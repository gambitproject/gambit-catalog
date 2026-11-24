import argparse
import hashlib
import io
import json
import re
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import pandas as pd
import pybtex.database
import pygambit as gbt
import yaml
from gtdraw import Style, pdf, png, svg, tex
from PIL import Image

CATALOG_RST_TABLE = Path(__file__).parent / "doc" / "_table.rst"
CATALOG_DIR = Path(__file__).parent / "games"
CATALOG_MANIFEST = Path(__file__).parent / "games" / "manifest.json"
MANIFEST_SCHEMA = Path(__file__).parent / "schema" / "manifest.schema.json"
VERSION_FILE = Path(__file__).parent / "VERSION"
REFERENCES_BIB = Path(__file__).parent / "references.bib"
SUPPORTED_GAME_FORMATS = {"efg", "nfg"}
GAME_READERS = {".efg": gbt.read_efg, ".nfg": gbt.read_nfg}
_CITE_RE = re.compile(r":cite:p:`([^`]+)`")
SCHEMA_VERSION = 1
SOURCE_FILE = "source.yaml"

HOUSE_STYLE = Style(
    color_scheme="colorblind",
    font_family="sffamily",
    font_italic=True,
    iset_fill=True,
    iset_boundary="none",
)
IMAGE_FORMATS = ("tex", "png", "pdf", "svg")
THUMBNAIL_SIZE = (560, 560)
THUMBNAIL_COLORS = 64


@dataclass
class Entry:
    """A catalog entry: a directory holding one game file and its drawings."""

    slug: str
    game_path: Path
    drawings: list[dict]

    @property
    def format(self) -> str:
        return self.game_path.suffix[1:]

    @property
    def source_path(self) -> Path:
        return self.game_path.parent / SOURCE_FILE


class CatalogError(ValueError):
    """The catalog's content is invalid; the message names the files concerned."""


def _entry_drawings(entry_dir: Path, stem: str, catalog_dir: Path) -> list[dict]:
    """Return the committed drawings in *entry_dir*, the primary drawing first.

    The primary drawing is ``{stem}.ef``; each additional drawing is
    ``{stem}__{suffix}.ef``, labelled by its title-cased suffix.
    """
    drawings = []
    primary = entry_dir / f"{stem}.ef"
    if primary.is_file():
        drawings.append(
            {
                "key": "default",
                "label": "Default",
                "path": primary.relative_to(catalog_dir).as_posix(),
            }
        )
    for ef_file in sorted(entry_dir.glob(f"{stem}__*.ef")):
        suffix = ef_file.stem[len(stem) + 2 :]  # noqa: E203
        drawings.append(
            {
                "key": suffix.lower(),
                "label": suffix.replace("_", " ").title(),
                "path": ef_file.relative_to(catalog_dir).as_posix(),
            }
        )
    return drawings


def discover_entries(catalog_dir: Path | None = None) -> list[Entry]:
    """Return the catalog's entries, sorted by slug.

    Each entry is a directory under *catalog_dir* whose path is the slug, holding
    exactly one game file named after the slug's last component.

    Raises
    ------
    ValueError
        If a game file is not named after its directory, or a directory holds
        more than one game file.
    """
    catalog_dir = catalog_dir or CATALOG_DIR
    img_dir = catalog_dir / "img"
    entries: dict[str, Entry] = {}
    for game_path in sorted(
        path
        for suffix in GAME_READERS
        for path in catalog_dir.rglob(f"*{suffix}")
        if path.is_file() and img_dir not in path.parents
    ):
        entry_dir = game_path.parent
        rel = game_path.relative_to(catalog_dir).as_posix()
        if entry_dir == catalog_dir or game_path.stem != entry_dir.name:
            raise ValueError(f"{rel}: a game file must be named after its directory")
        slug = entry_dir.relative_to(catalog_dir).as_posix()
        if slug in entries:
            raise ValueError(f"{slug}: an entry directory must hold exactly one game file")
        entries[slug] = Entry(
            slug=slug,
            game_path=game_path,
            drawings=_entry_drawings(entry_dir, game_path.stem, catalog_dir),
        )
    return [entries[slug] for slug in sorted(entries)]


def read_game(entry: Entry) -> gbt.Game:
    """Read the game file of *entry*."""
    return GAME_READERS[entry.game_path.suffix](str(entry.game_path))


def _catalog_games(catalog_dir: Path | None = None, bib_path: Path | None = None) -> pd.DataFrame:
    """Return one row per catalog entry: title, description, source, game file and drawings.

    Raises
    ------
    CatalogError
        If an entry's ``source.yaml`` is missing or invalid.
    """
    catalog_dir = catalog_dir or CATALOG_DIR
    entries = discover_entries(catalog_dir)
    sources, problems = _load_sources(entries, bib_path)
    if problems:
        raise CatalogError("\n".join(problems))
    records = []
    for entry in entries:
        game = read_game(entry)
        records.append(
            {
                "Game": entry.slug,
                "Title": game.title,
                "Description": game.description,
                "Format": entry.format,
                "Path": entry.game_path.relative_to(catalog_dir).as_posix(),
                "Drawings": entry.drawings,
                "Source": sources[entry.slug]["source"],
                "Name": sources[entry.slug].get("name"),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["Game", "Title", "Description", "Format", "Path", "Drawings", "Source", "Name"],
    )


def _images(slug: str, fmt: str, drawings: list[dict]) -> list[dict]:
    """Return the images made for an entry: one per drawing, or one for a strategic game.

    Each is a dict with ``key``, ``label``, ``source`` (the drawing's path under the catalog
    directory, or ``None`` for a strategic game) and ``base`` (the image path without
    extension, under the catalog directory).
    """
    if fmt == "nfg":
        return [
            {"key": "default", "label": "Default", "source": None, "base": f"img/{slug}/default"}
        ]
    return [
        {
            "key": d["key"],
            "label": d["label"],
            "source": d["path"],
            "base": f"img/{slug}/{d['key']}",
        }
        for d in drawings
    ]


def render_image(
    image: dict,
    game_path: Path,
    catalog_dir: Path,
    formats: tuple[str, ...] = IMAGE_FORMATS,
    regenerate: bool = False,
) -> None:
    """Compile *image* to each of *formats*, in the house style, unless already made.

    A drawing is compiled from its ``.ef`` file; a strategic game from its LaTeX table.
    """
    base = catalog_dir / image["base"]
    if not regenerate and all(base.with_suffix(f".{ext}").exists() for ext in formats):
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    if image["source"] is None:
        source = base.with_name(f"{base.name}-body.tex")
        source.write_text(gbt.read_nfg(str(game_path)).to_latex(), encoding="utf-8")
    else:
        source = catalog_dir / image["source"]
    for ext in formats:
        compile_to = {"tex": tex, "png": png, "pdf": pdf, "svg": svg}[ext]
        compile_to(str(source), save_to=str(base), style=HOUSE_STYLE)


def _warn_missing_descriptions(df: pd.DataFrame) -> None:
    """Print a warning to stderr for each game that lacks a description.

    Games without descriptions are excluded from the catalog page; this makes that visible
    so contributors know to add a description before the game will appear.
    """
    for _, row in df.iterrows():
        if str(row.get("Description", "")).strip():
            continue
        print(
            f"WARNING: '{row['Game']}' has no description and will not appear in the catalog.\n"
            f"Add a description to the game file to include it.",
            file=sys.stderr,
        )


def _entry_heading(row: pd.Series) -> str:
    """Return the heading of a game under its work: locator and name, else the game's title."""
    locator = row["Source"].get("locator")
    name = row.get("Name")
    name = name if isinstance(name, str) else None
    if locator and name:
        return f"{locator}: {name}"
    return locator or name or str(row.get("Title", "")).strip()


def _group_by_work(df: pd.DataFrame) -> list[tuple[dict, list[pd.Series]]]:
    """Group the games in *df* by source work.

    Works are sorted by citation, games within a work by slug. Games in an unsupported format
    or without a description are left out.
    """
    works: dict[str, tuple[dict, list[pd.Series]]] = {}
    for _, row in df.sort_values("Game").iterrows():
        if row.get("Format") not in SUPPORTED_GAME_FORMATS:
            continue
        if not str(row.get("Description", "")).strip():
            continue
        source = row["Source"]
        works.setdefault(source["key"], (source, []))[1].append(row)
    return sorted(works.values(), key=lambda w: (w[0]["citation"], w[0]["key"]))


def _write_game_entry(
    f,
    row: pd.Series,
    slug: str,
    catalog_dir: Path,
    indent: str,
    regenerate_images: bool = False,
) -> None:
    """Write RST for a single game entry (dropdown + content) at *indent* depth."""
    i0 = indent  # .. dropdown:: title line
    i1 = indent + "   "  # content inside game dropdown
    i2 = indent + "      "  # nested directives (Download, tab-set)
    i3 = indent + "         "  # content inside nested directives

    title = _entry_heading(row)
    description = str(row.get("Description", "")).strip()
    images = _images(slug, row["Format"], row["Drawings"])
    for image in images:
        render_image(image, catalog_dir / row["Path"], catalog_dir, regenerate=regenerate_images)

    f.write(f"{i0}.. dropdown:: {title}\n")
    f.write(f"{i0}   :open:\n")
    f.write(f"{i0}\n")
    for line in description.splitlines():
        f.write(f"{i1}{line}\n")
    f.write(f"{i1}\n")
    f.write(f"{i1}**Load in PyGambit:**\n")
    f.write(f"{i1}\n")
    f.write(f"{i1}.. code-block:: python\n")
    f.write(f"{i1}   \n")
    f.write(f'{i1}   pygambit.catalog.load("{slug}")\n')
    f.write(f"{i1}\n")

    download_links = [f":download:`{Path(row['Path']).name} <../games/{row['Path']}>`"]
    for d in row["Drawings"]:
        download_links.append(f":download:`{Path(d['path']).name} <../games/{d['path']}>`")
    for image in images:
        for ext in IMAGE_FORMATS:
            name = f"{Path(slug).name}-{image['key']}.{ext}"
            download_links.append(f":download:`{name} <../games/{image['base']}.{ext}>`")
    f.write(f"{i1}.. dropdown:: Download game and image files\n")
    f.write(f"{i1}   \n")
    f.write(f"{i2}{' '.join(download_links)}\n")
    f.write(f"{i1}\n")

    if len(images) > 1:
        f.write(f"{i1}.. tab-set::\n")
        f.write(f"{i1}\n")
        for image in images:
            f.write(f"{i2}.. tab-item:: {image['label']}\n")
            f.write(f"{i2}\n")
            f.write(f"{i3}.. image:: ../games/{image['base']}.svg\n")
            f.write(f"{i2}\n")
        f.write(f"{i1}\n")
    elif images:
        f.write(f"{i1}.. image:: ../games/{images[0]['base']}.svg\n")
        f.write(f"{i1}\n")


def generate_rst_table(
    df: pd.DataFrame,
    rst_path: Path,
    regenerate_images: bool = False,
    catalog_dir: Path | None = None,
):
    """Generate RST output with one dropdown per source work, holding that work's games."""
    catalog_dir = catalog_dir or CATALOG_DIR
    with open(rst_path, "w", encoding="utf-8") as f:
        for source, rows in _group_by_work(df):
            f.write(f".. dropdown:: {source['citation']}: {source['title']}\n\n")
            for row in rows:
                _write_game_entry(
                    f,
                    row,
                    row["Game"],
                    catalog_dir,
                    "   ",
                    regenerate_images=regenerate_images,
                )
            f.write("\n")


_TEX_LETTERS = {
    "o": "\u00f8",
    "O": "\u00d8",
    "ae": "\u00e6",
    "AE": "\u00c6",
    "aa": "\u00e5",
    "AA": "\u00c5",
    "ss": "\u00df",
    "l": "\u0142",
    "L": "\u0141",
    "i": "\u0131",
}
_TEX_ACCENTS = {
    '"': "\u0308",
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "c": "\u0327",
    "v": "\u030c",
}
_TEX_ACCENT_RE = re.compile(r"\\([\"'`^~=.]|[cv](?![A-Za-z]))\s*\{?\\?([A-Za-z])\}?")
_TEX_LETTER_RE = re.compile(
    r"\\(" + "|".join(sorted(_TEX_LETTERS, key=len, reverse=True)) + r")(?![A-Za-z])\s*"
)


def _plain(text: str) -> str:
    """Return *text* from a BibTeX field with LaTeX accents and braces turned into plain text."""
    text = _TEX_ACCENT_RE.sub(lambda m: m.group(2) + _TEX_ACCENTS[m.group(1)], text)
    text = _TEX_LETTER_RE.sub(lambda m: _TEX_LETTERS[m.group(1)], text)
    text = re.sub(r"\\([&%$#_])", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _bib_persons(entry: pybtex.database.Entry) -> list[pybtex.database.Person]:
    """Return the authors of *entry*, or its editors if it has none."""
    return list(entry.persons.get("author", [])) or list(entry.persons.get("editor", []))


def _surnames(entry: pybtex.database.Entry) -> list[str]:
    persons = _bib_persons(entry)
    return [_plain(" ".join(p.prelast_names + p.last_names)) for p in persons if p.last_names]


def _author_text(surnames: list[str]) -> str:
    """Return "Smith", "Smith and Jones", or "Smith et al." for a list of surnames."""
    if len(surnames) == 0:
        return ""
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}"
    return f"{surnames[0]} et al."


def _format_citation(entry: pybtex.database.Entry) -> str:
    """Return a short "(Author Year)" citation string for a single bibtex entry."""
    author_text = _author_text(_surnames(entry))
    year = entry.fields.get("year", "")
    if author_text and year:
        return f"({author_text} {year})"
    return f"({author_text or year})"


def short_citation(entry: pybtex.database.Entry) -> str:
    """Return the "Author (Year)" form of a bibtex entry's citation, as used in titles."""
    author_text = _author_text(_surnames(entry))
    year = entry.fields.get("year", "")
    return f"{author_text} ({year})"


def describe_work(entry: pybtex.database.Entry) -> dict:
    """Return the manifest's description of the work a bibtex entry stands for.

    The result has ``citation``, ``authors``, ``year``, ``title``, ``kind`` and, for an
    entry in a journal or a volume, ``venue``.

    Raises
    ------
    ValueError
        If the entry lacks an author or editor, a year or a title.
    """
    persons = _bib_persons(entry)
    year = entry.fields.get("year", "")
    title = _plain(entry.fields.get("title", ""))
    if not persons or not re.fullmatch(r"\d+", year.strip()) or not title:
        raise ValueError("needs an author or editor, a numeric year and a title")
    work = {
        "citation": short_citation(entry),
        "authors": [
            _plain(
                " ".join(
                    p.first_names
                    + p.middle_names
                    + p.prelast_names
                    + p.last_names
                    + p.lineage_names
                )
            )
            for p in persons
        ],
        "year": int(year),
        "title": title,
        "kind": entry.type.lower(),
    }
    venue = entry.fields.get("journal") or entry.fields.get("booktitle")
    if venue:
        work["venue"] = _plain(venue)
    return work


def load_citation_texts(bib_path: Path | None = None) -> dict[str, str]:
    """Return a mapping from bibtex citation key to a short "(Author Year)" citation string."""
    bib_path = bib_path or REFERENCES_BIB
    bib_data = pybtex.database.parse_file(str(bib_path), bib_format="bibtex")
    return {key: _format_citation(entry) for key, entry in bib_data.entries.items()}


def cited_keys(description: str) -> list[str]:
    """Return the citation keys of every ``:cite:p:`key``` in *description*, in order."""
    return _CITE_RE.findall(description)


def resolve_citations(description: str, citation_texts: dict[str, str]) -> str:
    """Replace every ``:cite:p:`key``` occurrence in *description* with a short citation string.

    Raises
    ------
    CatalogError
        If a cited key is not in *citation_texts*.
    """
    missing = [key for key in cited_keys(description) if key not in citation_texts]
    if missing:
        cited = ", ".join(map(repr, missing))
        raise CatalogError(f"description cites {cited}, not in references.bib")
    return _CITE_RE.sub(lambda m: citation_texts[m.group(1)], description)


def _check_mapping(path: Path, where: str, value, required: set, optional: set) -> None:
    if not isinstance(value, dict):
        raise CatalogError(f"{path}: '{where}' must be a mapping")
    for field in sorted(required - value.keys()):
        raise CatalogError(f"{path}: '{where}' needs a '{field}'")
    for field in sorted(value.keys() - required - optional, key=str):
        raise CatalogError(f"{path}: unknown key '{field}' in '{where}'")
    for field, text in value.items():
        if not isinstance(text, str) or not text.strip():
            raise CatalogError(f"{path}: '{where}.{field}' must be a non-empty string")


def read_source_file(path: Path) -> dict:
    """Read and validate a ``source.yaml`` file.

    Returns
    -------
    dict
        With ``source`` (``key`` and optional ``locator``) and, when present, ``origin``
        (``key``, optional ``locator`` and ``note``) and ``name``.

    Raises
    ------
    CatalogError
        If the file is not valid YAML, lacks ``source``, or has unknown keys or values of
        the wrong type; the message names the file.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CatalogError(f"{path}: not valid YAML ({e})") from e
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: must be a mapping with a 'source'")
    if "source" not in data:
        raise CatalogError(f"{path}: needs a 'source'")
    unknown = sorted(data.keys() - {"source", "origin", "name"}, key=str)
    if unknown:
        raise CatalogError(f"{path}: unknown key '{unknown[0]}'")
    _check_mapping(path, "source", data["source"], {"key"}, {"locator"})
    if "origin" in data:
        _check_mapping(path, "origin", data["origin"], {"key"}, {"locator", "note"})
    if "name" in data and (not isinstance(data["name"], str) or not data["name"].strip()):
        raise CatalogError(f"{path}: 'name' must be a non-empty string")
    return data


def load_references(bib_path: Path | None = None) -> dict[str, pybtex.database.Entry]:
    """Return the bibliography's entries by citation key."""
    bib_path = bib_path or REFERENCES_BIB
    return dict(pybtex.database.parse_file(str(bib_path), bib_format="bibtex").entries)


def _resolve_work(path: Path, where: str, block: dict, refs: dict) -> dict:
    key = block["key"]
    if key not in refs:
        raise CatalogError(f"{path}: '{where}.key' {key!r} is not in references.bib")
    try:
        work = describe_work(refs[key])
    except ValueError as e:
        raise CatalogError(f"references.bib entry {key!r} (used by {path}) {e}") from e
    resolved = {"key": key}
    if "locator" in block:
        resolved["locator"] = block["locator"]
    resolved.update(work)
    if "note" in block:
        resolved["note"] = block["note"]
    return resolved


def resolve_source(path: Path, refs: dict) -> dict:
    """Read ``source.yaml`` at *path* and resolve its keys against *refs*.

    Returns a dict with ``source`` and, when present, ``origin`` and ``name``; each work
    is described as in the manifest.

    Raises
    ------
    CatalogError
        If the file is missing or invalid, or a key is not in the bibliography.
    """
    if not path.is_file():
        raise CatalogError(f"{path}: missing")
    data = read_source_file(path)
    resolved = {"source": _resolve_work(path, "source", data["source"], refs)}
    if "origin" in data:
        resolved["origin"] = _resolve_work(path, "origin", data["origin"], refs)
    if "name" in data:
        resolved["name"] = data["name"].strip()
    return resolved


def derived_title(resolved: dict) -> str:
    """Return the title a game must carry: ``<citation>[, <locator>][: <name>]``.

    *resolved* is the result of :func:`resolve_source`.
    """
    source = resolved["source"]
    title = source["citation"]
    if "locator" in source:
        title += f", {source['locator']}"
    if "name" in resolved:
        title += f": {resolved['name']}"
    return title


def _load_sources(
    entries: list[Entry], bib_path: Path | None = None
) -> tuple[dict[str, dict], list[str]]:
    """Resolve every entry's ``source.yaml``; return the results by slug and the problems found."""
    refs = load_references(bib_path)
    sources, problems = {}, []
    for entry in entries:
        try:
            sources[entry.slug] = resolve_source(entry.source_path, refs)
        except CatalogError as e:
            problems.append(str(e))
    return sources, problems


def catalog_problems(catalog_dir: Path | None = None, bib_path: Path | None = None) -> list[str]:
    """Return every problem in the catalog's sources and titles, one message each.

    A problem is a missing or invalid ``source.yaml``, a key in it or in the game's
    description that is absent from the bibliography, or a game whose title differs from the
    one derived from its source.
    """
    entries = discover_entries(catalog_dir or CATALOG_DIR)
    sources, problems = _load_sources(entries, bib_path)
    refs = load_references(bib_path)
    for entry in entries:
        game = read_game(entry)
        for key in cited_keys(str(game.description)):
            if key not in refs:
                problems.append(f"{entry.slug}: description cites {key!r}, not in references.bib")
        if entry.slug not in sources:
            continue
        expected = derived_title(sources[entry.slug])
        if game.title != expected:
            problems.append(f"{entry.slug}: title is {game.title!r}, expected {expected!r}")
    return problems


_HEADER_RE = re.compile(r'\A((?:EFG|NFG) \d+ [A-Za-z]+ )"((?:[^"\\\n]|\\.)*)"')


def _set_header_title(text: str, title: str) -> str:
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    new, n = _HEADER_RE.subn(lambda m: f'{m.group(1)}"{escaped}"', text, count=1)
    if n != 1:
        raise CatalogError("the header line does not have the expected form")
    return new


def sync_titles(
    catalog_dir: Path | None = None, bib_path: Path | None = None, write: bool = True
) -> list[str]:
    """Set each game's title to the one derived from its source; return the slugs that differed.

    Only the title string in the file's header line is changed; with ``write=False`` nothing
    is written.

    Raises
    ------
    CatalogError
        If any entry's ``source.yaml`` is missing or invalid.
    """
    entries = discover_entries(catalog_dir or CATALOG_DIR)
    sources, problems = _load_sources(entries, bib_path)
    if problems:
        raise CatalogError("\n".join(problems))
    changed = []
    for entry in entries:
        expected = derived_title(sources[entry.slug])
        if read_game(entry).title == expected:
            continue
        changed.append(entry.slug)
        if write:
            text = entry.game_path.read_bytes().decode("utf-8")
            try:
                text = _set_header_title(text, expected)
            except CatalogError as e:
                raise CatalogError(f"{entry.game_path}: {e}") from e
            entry.game_path.write_bytes(text.encode("utf-8"))
    return changed


def _game_stats(game: gbt.Game) -> dict:
    """Return a subset of *game*'s structural attributes, for catalog-browser filtering."""
    return {
        "n_players": len(game.players),
        "is_const_sum": game.is_const_sum(),
        "n_strategies": sum(len(game.get_strategies(p)) for p in game.players),
    }


def catalog_version(version_file: Path | None = None) -> str:
    """Return the catalog's version, from the ``VERSION`` file."""
    return (version_file or VERSION_FILE).read_text(encoding="utf-8").strip()


def build_manifest(
    catalog_dir: Path | None = None,
    bib_path: Path | None = None,
    version: str | None = None,
) -> dict:
    """Build the catalog manifest, the index of the catalog's games.

    One entry per game, with its game file and drawings, its source (and origin and name,
    where it has them), a citation-resolved plain-text description, and a subset of
    structural stats. Does not require gtdraw or a LaTeX toolchain: unlike
    :func:`generate_rst_table`, it never renders an image.

    Raises
    ------
    CatalogError
        If any game lacks a valid ``source.yaml`` or has a title other than the derived one;
        the message lists every offender.
    """
    catalog_dir = catalog_dir or CATALOG_DIR
    version = version or catalog_version()
    problems = catalog_problems(catalog_dir, bib_path)
    if problems:
        raise CatalogError("\n".join(problems))
    citation_texts = load_citation_texts(bib_path)
    refs = load_references(bib_path)

    games = []
    for entry in discover_entries(catalog_dir):
        game = read_game(entry)
        description = str(game.description).strip()
        if not description:
            continue
        resolved = resolve_source(entry.source_path, refs)
        record = {
            "slug": entry.slug,
            "title": game.title,
            **({"name": resolved["name"]} if "name" in resolved else {}),
            "description": resolve_citations(description, citation_texts),
            "format": entry.format,
            "game": entry.game_path.relative_to(catalog_dir).as_posix(),
            "drawings": entry.drawings,
            "source": resolved["source"],
            **({"origin": resolved["origin"]} if "origin" in resolved else {}),
            **_game_stats(game),
        }
        games.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": version,
        "games": games,
    }


def validate_manifest(manifest: dict, schema_path: Path | None = None) -> None:
    """Validate *manifest* against the manifest JSON Schema.

    Raises
    ------
    jsonschema.ValidationError
        If the manifest does not conform to the schema.
    """
    with open(schema_path or MANIFEST_SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator(schema).validate(manifest)


def write_manifest(manifest: dict, manifest_path: Path | None = None) -> None:
    """Write *manifest* to *manifest_path* as JSON, only touching the file if content changed."""
    manifest_path = manifest_path or CATALOG_MANIFEST
    updated_content = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""

    if content != updated_content:
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated {str(manifest_path)}")
    else:
        print(f"No changes to add to {str(manifest_path)}")


def make_thumbnail(png_path: Path) -> bytes:
    """Return a PNG thumbnail of the image at *png_path*, at most ``THUMBNAIL_SIZE``.

    The thumbnail has a palette of ``THUMBNAIL_COLORS`` colours, which suits line drawings.
    """
    with Image.open(png_path) as im:
        im.thumbnail(THUMBNAIL_SIZE)
        thumb = im.convert("RGB").quantize(colors=THUMBNAIL_COLORS)
        out = io.BytesIO()
        thumb.save(out, format="PNG", optimize=True)
    return out.getvalue()


RELEASE_FILES = ("LICENSE", "LICENSES", "README.md", "schema/manifest.schema.json")
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def build_release(
    outdir: Path,
    catalog_dir: Path | None = None,
    regenerate_images: bool = False,
) -> Path:
    """Write the release archive ``gambit-catalog-<version>.zip`` and its ``.sha256`` to *outdir*.

    The archive holds, under ``gambit-catalog-<version>/``, the committed manifest
    (``games/manifest.json``), every game file and drawing it lists, the licences, the
    README, the manifest's schema, and a thumbnail ``thumbs/<slug>.png`` for every game.

    Returns
    -------
    Path
        The path of the archive.

    Raises
    ------
    RuntimeError
        If the committed manifest is out of date, or a thumbnail cannot be made; the
        message names the games concerned.
    """
    catalog_dir = catalog_dir or CATALOG_DIR
    root = catalog_dir.parent
    manifest_path = catalog_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest != build_manifest(catalog_dir, version=manifest["catalog_version"]):
        raise RuntimeError(
            "games/manifest.json is out of date; run 'python build.py --build' and commit it"
        )
    version = manifest["catalog_version"]

    thumbs: dict[str, bytes] = {}
    failed = []
    for game in manifest["games"]:
        try:
            image = _images(game["slug"], game["format"], game["drawings"])[0]
            render_image(
                image,
                catalog_dir / game["game"],
                catalog_dir,
                formats=("png",),
                regenerate=regenerate_images,
            )
            thumbs[game["slug"]] = make_thumbnail(catalog_dir / f"{image['base']}.png")
        except Exception as e:  # noqa: BLE001 - every failure is reported by slug below
            failed.append(f"{game['slug']} ({e})")
    if failed:
        raise RuntimeError("cannot make a thumbnail for: " + "; ".join(failed))

    members: dict[str, bytes] = {"games/manifest.json": manifest_path.read_bytes()}
    for game in manifest["games"]:
        for rel in [game["game"], *(d["path"] for d in game["drawings"])]:
            members[f"games/{rel}"] = (catalog_dir / rel).read_bytes()
        members[f"thumbs/{game['slug']}.png"] = thumbs[game["slug"]]
    for name in RELEASE_FILES:
        path = root / name
        for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if file.is_file():
                members[file.relative_to(root).as_posix()] = file.read_bytes()

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"gambit-catalog-{version}"
    archive = outdir / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(f"{stem}/{name}", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (outdir / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Update the Gambit games catalog's documentation and manifest. "
            "Regenerates doc/_table.rst from the current catalog by default; "
            "pass --skip-table to skip this (it requires a LaTeX toolchain via gtdraw)."
        )
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "Also regenerate games/manifest.json, the catalog manifest. "
            "Required after adding, removing, or editing games."
        ),
    )
    parser.add_argument(
        "--sync-titles",
        action="store_true",
        help=(
            "Rewrite each game file's title to the one derived from its source.yaml, "
            "changing nothing else in the file."
        ),
    )
    parser.add_argument(
        "--check-titles",
        action="store_true",
        help="List the games whose title differs from the derived one, and exit.",
    )
    parser.add_argument(
        "--regenerate-images",
        action="store_true",
        help=(
            "Force regeneration of all game visualisation images (PNG, PDF, SVG, TeX), "
            "even if they already exist. Use this to pick up changes to game files or "
            "drawings."
        ),
    )
    parser.add_argument(
        "--skip-table",
        action="store_true",
        help=(
            "Skip regenerating doc/_table.rst and any missing game images. "
            "Table/image generation requires a LaTeX toolchain (gtdraw's tex/pdf/png/svg "
            "functions); use this with --build in contexts (e.g. CI) that only need "
            "the manifest checked or updated."
        ),
    )
    parser.add_argument(
        "--release",
        metavar="OUTDIR",
        type=Path,
        help=(
            "Write the release archive gambit-catalog-<VERSION>.zip and its .sha256 to "
            "OUTDIR. Requires a LaTeX toolchain via gtdraw, for the thumbnails."
        ),
    )
    args = parser.parse_args()

    try:
        if args.check_titles:
            problems = catalog_problems()
            if problems:
                sys.exit("\n".join(problems))
            print("All titles are as derived from the sources.")
            sys.exit(0)
        if args.sync_titles:
            changed = sync_titles()
            print(f"Updated the titles of {len(changed)} game file(s).")
            for slug in changed:
                print(f"  {slug}")
        if args.build or not args.skip_table:
            problems = catalog_problems()
            if problems:
                sys.exit("Cannot build the catalog:\n" + "\n".join(problems))
        if not args.skip_table:
            # Create RST list-table used by doc/index.rst
            df = _catalog_games()
            _warn_missing_descriptions(df)
            generate_rst_table(df, CATALOG_RST_TABLE, regenerate_images=args.regenerate_images)
            print(f"Generated {CATALOG_RST_TABLE} for use in local docs build. DO NOT COMMIT.")
        if args.build:
            manifest = build_manifest()
            validate_manifest(manifest)
            write_manifest(manifest)
    except CatalogError as e:
        sys.exit(str(e))
    if args.release:
        archive = build_release(args.release, regenerate_images=args.regenerate_images)
        print(f"Wrote {archive} and {archive.name}.sha256")
