import argparse
import hashlib
import io
import json
import re
import sys
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
CATALOG_HIERARCHY_CONFIG = Path(__file__).parent / "hierarchy.yaml"
REFERENCES_BIB = Path(__file__).parent / "references.bib"
SUPPORTED_GAME_FORMATS = {"efg", "nfg"}
GAME_READERS = {".efg": gbt.read_efg, ".nfg": gbt.read_nfg}
_CITE_RE = re.compile(r":cite:p:`([^`]+)`")

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


def _catalog_games(catalog_dir: Path | None = None) -> pd.DataFrame:
    """Return one row per catalog entry, with its title, description, game file and drawings."""
    catalog_dir = catalog_dir or CATALOG_DIR
    records = []
    for entry in discover_entries(catalog_dir):
        game = read_game(entry)
        records.append(
            {
                "Game": entry.slug,
                "Title": game.title,
                "Description": game.description,
                "Format": entry.format,
                "Path": entry.game_path.relative_to(catalog_dir).as_posix(),
                "Drawings": entry.drawings,
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["Game", "Title", "Description", "Format", "Path", "Drawings"]
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


def load_hierarchy_labels() -> dict[str, str]:
    """Return the human-readable label mapping for catalog hierarchy nodes."""
    with open(CATALOG_HIERARCHY_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("labels", {})


def _node_label(prefix: str, labels: dict[str, str]) -> str:
    """Return the display label for a catalog hierarchy node.

    Falls back to title-casing the last path component when the prefix is not
    listed in the YAML config.
    """
    if prefix in labels:
        return labels[prefix]
    return Path(prefix).name.replace("_", " ").title()


def _warn_missing_descriptions(df: pd.DataFrame) -> None:
    """Print a warning to stderr for each game that lacks a description.

    Games without descriptions are silently excluded from the catalog page by
    ``_build_slug_tree``.  This function makes that visible so contributors
    know to add a description before the game will appear.
    """
    for _, row in df.iterrows():
        if str(row.get("Description", "")).strip():
            continue
        print(
            f"WARNING: '{row['Game']}' has no description and will not appear in the catalog.\n"
            f"Add a description to the game file to include it.",
            file=sys.stderr,
        )


def _build_slug_tree(df: pd.DataFrame) -> dict:
    """Build a nested dict tree from the slugs in *df*.

    Intermediate nodes are plain ``dict``s.  Leaf nodes (individual games) store
    the corresponding ``pd.Series`` row.  Slugs are split on ``"/"`` to produce
    the nesting.
    """
    tree: dict = {}
    for _, row in df.iterrows():
        if row.get("Format") not in SUPPORTED_GAME_FORMATS:
            continue
        if not str(row.get("Description", "")).strip():
            continue
        parts = row["Game"].split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = row
    return tree


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

    title = str(row.get("Title", "")).strip()
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


def _write_tree_level(
    f,
    subtree: dict,
    path_prefix: str,
    labels: dict[str, str],
    catalog_dir: Path,
    indent: str,
    regenerate_images: bool = False,
) -> None:
    """Recursively write RST nested dropdowns for *subtree*.

    Intermediate nodes (``dict`` values) become collapsible ``.. dropdown::``
    sections.  Top-level nodes (those whose ``path_prefix`` contains no ``/``)
    are rendered open by default so the first level of the hierarchy is
    immediately visible.  Leaf nodes (``pd.Series`` values) are individual
    games rendered via :func:`_write_game_entry`.
    """
    for key in sorted(subtree):
        value = subtree[key]
        child_prefix = f"{path_prefix}/{key}" if path_prefix else key

        if isinstance(value, dict):
            label = _node_label(child_prefix, labels)
            is_top_level = "/" not in child_prefix
            f.write(f"{indent}.. dropdown:: {label}\n")
            if is_top_level:
                f.write(f"{indent}   :open:\n")
            f.write(f"{indent}\n")
            _write_tree_level(
                f,
                value,
                child_prefix,
                labels,
                catalog_dir,
                indent + "   ",
                regenerate_images=regenerate_images,
            )
            f.write(f"{indent}\n")
        else:
            _write_game_entry(
                f,
                value,
                child_prefix,
                catalog_dir,
                indent,
                regenerate_images=regenerate_images,
            )


def generate_rst_table(
    df: pd.DataFrame,
    rst_path: Path,
    regenerate_images: bool = False,
    catalog_dir: Path | None = None,
):
    """Generate RST output with nested dropdowns grouped by catalog hierarchy."""
    catalog_dir = catalog_dir or CATALOG_DIR
    labels = load_hierarchy_labels()
    tree = _build_slug_tree(df)
    with open(rst_path, "w", encoding="utf-8") as f:
        _write_tree_level(
            f, tree, "", labels, catalog_dir, indent="", regenerate_images=regenerate_images
        )


def _format_citation(entry: pybtex.database.Entry) -> str:
    """Return a short "(Author Year)" citation string for a single bibtex entry."""
    persons = entry.persons.get("author", [])
    last_names = [
        " ".join(str(part) for part in p.prelast_names + p.last_names)
        for p in persons
        if p.last_names
    ]
    if len(last_names) == 0:
        author_text = ""
    elif len(last_names) == 1:
        author_text = last_names[0]
    elif len(last_names) == 2:
        author_text = f"{last_names[0]} and {last_names[1]}"
    else:
        author_text = f"{last_names[0]} et al."
    year = entry.fields.get("year", "")
    if author_text and year:
        return f"({author_text} {year})"
    return f"({author_text or year})"


def load_citation_texts(bib_path: Path | None = None) -> dict[str, str]:
    """Return a mapping from bibtex citation key to a short "(Author Year)" citation string."""
    bib_path = bib_path or REFERENCES_BIB
    bib_data = pybtex.database.parse_file(str(bib_path), bib_format="bibtex")
    return {key: _format_citation(entry) for key, entry in bib_data.entries.items()}


def resolve_citations(description: str, citation_texts: dict[str, str]) -> str:
    """Replace every ``:cite:p:`key``` occurrence in *description* with a short citation string.

    Falls back to ``(key)`` for a key not found in *citation_texts*, so a manifest build
    never silently drops a citation just because the bib entry couldn't be resolved.
    """
    return _CITE_RE.sub(lambda m: citation_texts.get(m.group(1), f"({m.group(1)})"), description)


def _hierarchy_breadcrumbs(slug: str, labels: dict[str, str]) -> list[str]:
    """Return the human-readable label for each ancestor group of *slug*, root to leaf."""
    breadcrumbs = []
    prefix = ""
    for part in slug.split("/")[:-1]:
        prefix = f"{prefix}/{part}" if prefix else part
        breadcrumbs.append(_node_label(prefix, labels))
    return breadcrumbs


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

    One entry per game, with its game file and drawings, a citation-resolved plain-text
    description, hierarchy breadcrumbs (for display/filtering), and a subset of structural
    stats. Does not require gtdraw or a LaTeX toolchain — unlike :func:`generate_rst_table`,
    it never renders an image.
    """
    catalog_dir = catalog_dir or CATALOG_DIR
    version = version or catalog_version()
    labels = load_hierarchy_labels()
    citation_texts = load_citation_texts(bib_path)

    games = []
    for entry in discover_entries(catalog_dir):
        game = read_game(entry)
        description = str(game.description).strip()
        if not description:
            continue
        breadcrumbs = _hierarchy_breadcrumbs(entry.slug, labels)
        games.append(
            {
                "slug": entry.slug,
                "title": game.title,
                "description": resolve_citations(description, citation_texts),
                "format": entry.format,
                "game": entry.game_path.relative_to(catalog_dir).as_posix(),
                "drawings": entry.drawings,
                "category": breadcrumbs[0] if breadcrumbs else "",
                "group": breadcrumbs[-1] if breadcrumbs else "",
                **_game_stats(game),
            }
        )
    return {
        "schema_version": int(version.split(".")[0]),
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
    if args.release:
        archive = build_release(args.release, regenerate_images=args.regenerate_images)
        print(f"Wrote {archive} and {archive.name}.sha256")
