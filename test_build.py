"""Tests for build.py.

All catalog slugs used here are clearly fictional (e.g. ``"testgroup2000/fig1"``)
and do not correspond to any game in the real catalog.  This is intentional: the
tests construct their own temporary catalog directories and DataFrame rows so they
are completely isolated from the actual catalog on disk.

Monkeypatching strategy
-----------------------
``build.py`` depends on two external resources that are replaced in tests:

1. ``tex`` / ``png`` / ``pdf`` / ``svg``
   (functions imported from ``gtdraw``) — replaced with no-ops, call-tracking
   lambdas, or a stub that writes a small PNG.  This lets us test RST and release
   generation without invoking LaTeX, and assert whether image generation was
   triggered at all.

2. ``catalog_dir`` (an argument to ``generate_rst_table`` and ``build_release``) —
   tests pass a ``tmp_path``-based directory, keeping all file I/O inside pytest's
   temporary directory and avoiding any reads from or writes to the repo.
"""

import hashlib
import io
import json
import textwrap
import zipfile

import jsonschema
import pytest

pytest.importorskip("gtdraw")  # build.py imports gtdraw at module level
pytest.importorskip("yaml")

import pandas as pd  # noqa: E402

import build  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level test fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _write_yaml(path, content):
    """Write *content* to *path* and return *path*."""
    path.write_text(content, encoding="utf-8")
    return path


def _write_efg_game(path, title="Test Game", description=""):
    """Write a minimal, valid, 2-player extensive-form game file to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'EFG 2 R "{title}" {{ "P1" "P2" }}\n'
        f'"{description}"\n'
        'p "" 1 1 "" { "l" "r" } 0\n'
        't "" 1 "" { 1, -1 }\n'
        't "" 2 "" { 2, -2 }\n',
        encoding="utf-8",
    )
    return path


def _write_nfg_game(path, title="Test NFG Game", description=""):
    """Write a minimal, valid, 2-player 2x2 normal-form game file to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'NFG 1 R "{title}" {{ "P1" "P2" }}\n\n'
        '{ { "1" "2" }\n{ "1" "2" }\n}\n'
        f'"{description}"\n\n'
        '{\n{ "" 1, 1 }\n{ "" 0, 0 }\n{ "" 0, 0 }\n{ "" 1, 1 }\n}\n'
        "1 2 3 4\n",
        encoding="utf-8",
    )
    return path


_TEST_BIB = textwrap.dedent("""\
    @article{Solo2020,
      author = {Solo, A.},
      title = {A {Solo} Paper},
      journal = {Journal of Tests},
      year = {2020},
    }
    @article{Duo2019,
      author = {First, A. and Second, B.},
      title = {Two Authors},
      journal = {Journal of Tests},
      year = {2019},
    }
    @article{Trio2018,
      author = {One, A. and Two, B. and Three, C.},
      title = {Three Authors},
      journal = {Journal of Tests},
      year = {2018},
    }
    @article{VonParticle2021,
      author = {von Particle, X. Y.},
      title = {A Particle},
      journal = {Journal of Tests},
      year = {2021},
    }
    @book{Book2010,
      author = {Writer, W.},
      title = {The Book},
      publisher = {Press},
      year = {2010},
    }
    @inproceedings{Conf2016,
      author = {S{\\o}rensen, T. B. and Other, O. and Third, T.},
      title = {Proceedings Paper: \\"Uber Games},
      booktitle = {Proceedings of the Test Conference},
      year = {2016},
    }
    @book{Edited2005,
      editor = {Editor, E.},
      title = {Edited Volume},
      publisher = {Press},
      year = {2005},
    }
    @article{NoYear,
      author = {Nobody, N.},
      title = {Undated},
    }
""")

# Titles of the fixture games' works, as derived from the bibliography above.
_CITATIONS = {"Solo2020": "Solo (2020)", "Duo2019": "First and Second (2019)"}


def _write_bib(path, content=_TEST_BIB):
    path.write_text(content, encoding="utf-8")
    return path


def _write_source(entry_dir, key="Solo2020", locator=None, name=None, origin=None, extra=""):
    """Write a ``source.yaml`` into *entry_dir*."""
    lines = ["source:", f"  key: {key}"]
    if locator:
        lines.append(f"  locator: {locator}")
    if origin:
        lines.append("origin:")
        lines.extend(f"  {k}: {v}" for k, v in origin.items())
    if name:
        lines.append(f"name: {name}")
    entry_dir.mkdir(parents=True, exist_ok=True)
    path = entry_dir / "source.yaml"
    path.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return path


def _add_game(
    catalog_dir,
    slug,
    key="Solo2020",
    locator=None,
    name=None,
    origin=None,
    title=None,
    description="A description.",
    fmt="efg",
):
    """Add an entry with a source.yaml; its title is the derived one unless *title* is given."""
    entry_dir = catalog_dir / slug
    _write_source(entry_dir, key, locator, name, origin)
    if title is None:
        title = (
            _CITATIONS[key] + (f", {locator}" if locator else "") + (f": {name}" if name else "")
        )
    writer = _write_efg_game if fmt == "efg" else _write_nfg_game
    return writer(
        entry_dir / f"{slug.rsplit('/', 1)[-1]}.{fmt}", title=title, description=description
    )


_FAKE_SOURCE = {
    "key": "Fake2000",
    "citation": "Fake (2000)",
    "authors": ["A. Fake"],
    "year": 2000,
    "title": "A Fake Work",
    "kind": "book",
}


def _efg_row(
    slug,
    title="Test EFG Game",
    description="A description.",
    drawings=None,
    source=None,
    name=None,
):
    """Return a dict representing one row of the DataFrame produced by
    ``build._catalog_games`` for an extensive-form game.

    Unless *drawings* is given, the entry has a primary drawing only.
    """
    stem = slug.rsplit("/", 1)[-1]
    if drawings is None:
        drawings = [{"key": "default", "label": "Default", "path": f"{slug}/{stem}.ef"}]
    return {
        "Game": slug,
        "Title": title,
        "Description": description,
        "Format": "efg",
        "Path": f"{slug}/{stem}.efg",
        "Drawings": drawings,
        "Source": source or _FAKE_SOURCE,
        "Name": name,
    }


def _nfg_row(slug, title="Test NFG Game", description="A description.", source=None, name=None):
    """Return a dict representing one row of the DataFrame for a normal-form game."""
    stem = slug.rsplit("/", 1)[-1]
    return {
        "Game": slug,
        "Title": title,
        "Description": description,
        "Format": "nfg",
        "Path": f"{slug}/{stem}.nfg",
        "Drawings": [],
        "Source": source or _FAKE_SOURCE,
        "Name": name,
    }


def _make_df(*rows):
    """Build a DataFrame from one or more row dicts as ``generate_rst_table`` expects."""
    return pd.DataFrame(list(rows))


def _make_image_files(catalog_dir, slug, keys=("default",)):
    """Create stub image files under *catalog_dir*/img/*slug*/ for each image key.

    ``render_image`` skips images whose files all exist, so tests that are not
    about image generation can use this to set up the pre-existing-images state.
    """
    img_dir = catalog_dir / "img" / slug
    img_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        for ext in ["tex", "png", "pdf", "svg"]:
            (img_dir / f"{key}.{ext}").touch()


def _record_calls(monkeypatch):
    """Replace the gtdraw compile functions with ones that record (function, source, save_to)."""
    calls = []
    for name in ["tex", "png", "pdf", "svg"]:
        monkeypatch.setattr(
            build,
            name,
            lambda source, save_to, style, _n=name: calls.append((_n, source, save_to)),
        )
    return calls


def _fake_png(source, save_to, style):
    """Stand-in for ``gtdraw.png``: write a small PNG image to *save_to*.png."""
    from PIL import Image

    Image.new("RGB", (1200, 600), "white").save(f"{save_to}.png")


# ---------------------------------------------------------------------------
# Tests for discover_entries and _catalog_games
# ---------------------------------------------------------------------------


@pytest.mark.catalog_update
class TestDiscoverEntries:
    """Tests for ``discover_entries(catalog_dir)``: one directory per entry."""

    def test_entries_found_by_directory(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(catalog_dir / "src2000" / "fig1" / "fig1.efg")
        _write_nfg_game(catalog_dir / "src2000" / "tab1" / "tab1.nfg")
        entries = build.discover_entries(catalog_dir)
        assert [e.slug for e in entries] == ["src2000/fig1", "src2000/tab1"]
        assert [e.format for e in entries] == ["efg", "nfg"]

    def test_drawings_listed_primary_first(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        entry_dir = catalog_dir / "src2000" / "fig1"
        _write_efg_game(entry_dir / "fig1.efg")
        (entry_dir / "fig1__very_wide.ef").touch()
        (entry_dir / "fig1.ef").touch()
        (entry,) = build.discover_entries(catalog_dir)
        assert entry.drawings == [
            {"key": "default", "label": "Default", "path": "src2000/fig1/fig1.ef"},
            {
                "key": "very_wide",
                "label": "Very Wide",
                "path": "src2000/fig1/fig1__very_wide.ef",
            },
        ]

    def test_generated_images_ignored(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(catalog_dir / "img" / "src2000" / "fig1" / "fig1.efg")
        assert build.discover_entries(catalog_dir) == []

    def test_game_not_named_after_directory_rejected(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(catalog_dir / "src2000" / "fig1.efg")
        with pytest.raises(ValueError, match="named after its directory"):
            build.discover_entries(catalog_dir)

    def test_two_game_files_in_one_directory_rejected(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(catalog_dir / "src2000" / "fig1" / "fig1.efg")
        _write_nfg_game(catalog_dir / "src2000" / "fig1" / "fig1.nfg")
        with pytest.raises(ValueError, match="exactly one game file"):
            build.discover_entries(catalog_dir)

    def test_catalog_games_reads_title_description_and_source(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(
            catalog_dir, "src2000/fig1", locator="Figure 1", name="The game", description="D"
        )
        df = build._catalog_games(catalog_dir, _write_bib(tmp_path / "refs.bib"))
        (row,) = df.to_dict("records")
        assert row["Game"] == "src2000/fig1"
        assert row["Title"] == "Solo (2020), Figure 1: The game"
        assert row["Description"] == "D"
        assert row["Source"]["key"] == "Solo2020"
        assert row["Source"]["locator"] == "Figure 1"
        assert row["Name"] == "The game"

    def test_catalog_games_fails_without_source(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(catalog_dir / "src2000" / "fig1" / "fig1.efg")
        with pytest.raises(build.CatalogError, match="source.yaml: missing"):
            build._catalog_games(catalog_dir, _write_bib(tmp_path / "refs.bib"))


# ---------------------------------------------------------------------------
# Tests for render_image
# ---------------------------------------------------------------------------


@pytest.mark.catalog_update
class TestRenderImage:
    """Tests for ``_images`` and ``render_image``."""

    def test_images_one_per_drawing(self):
        drawings = [
            {"key": "default", "label": "Default", "path": "a/fig1/fig1.ef"},
            {"key": "wide", "label": "Wide", "path": "a/fig1/fig1__wide.ef"},
        ]
        images = build._images("a/fig1", "efg", drawings)
        assert [(i["key"], i["source"], i["base"]) for i in images] == [
            ("default", "a/fig1/fig1.ef", "img/a/fig1/default"),
            ("wide", "a/fig1/fig1__wide.ef", "img/a/fig1/wide"),
        ]

    def test_strategic_game_has_one_image(self):
        (image,) = build._images("a/tab1", "nfg", [])
        assert image["source"] is None
        assert image["base"] == "img/a/tab1/default"

    def test_drawing_compiled_in_house_style(self, tmp_path, monkeypatch):
        seen = []
        for name in ["tex", "png", "pdf", "svg"]:
            monkeypatch.setattr(
                build, name, lambda source, save_to, style: seen.append((source, style))
            )
        (image,) = build._images("a/fig1", "efg", _efg_row("a/fig1")["Drawings"])
        build.render_image(image, tmp_path / "a/fig1/fig1.efg", tmp_path)
        assert len(seen) == 4
        assert {src for src, _ in seen} == {str(tmp_path / "a/fig1/fig1.ef")}
        assert all(style == build.HOUSE_STYLE for _, style in seen)

    def test_strategic_game_compiled_from_its_latex(self, tmp_path, monkeypatch):
        calls = _record_calls(monkeypatch)
        game_path = _write_nfg_game(tmp_path / "a/tab1/tab1.nfg", title="Table")
        (image,) = build._images("a/tab1", "nfg", [])
        build.render_image(image, game_path, tmp_path, formats=("png",))
        body = tmp_path / "img/a/tab1/default-body.tex"
        assert calls == [("png", str(body), str(tmp_path / "img/a/tab1/default"))]
        assert "\\begin{game}" in body.read_text()

    def test_not_regenerated_when_all_exist(self, tmp_path, monkeypatch):
        calls = _record_calls(monkeypatch)
        _make_image_files(tmp_path, "a/fig1")
        (image,) = build._images("a/fig1", "efg", _efg_row("a/fig1")["Drawings"])
        build.render_image(image, tmp_path / "a/fig1/fig1.efg", tmp_path)
        assert calls == []

    def test_regenerated_when_flag_set(self, tmp_path, monkeypatch):
        calls = _record_calls(monkeypatch)
        _make_image_files(tmp_path, "a/fig1")
        (image,) = build._images("a/fig1", "efg", _efg_row("a/fig1")["Drawings"])
        build.render_image(image, tmp_path / "a/fig1/fig1.efg", tmp_path, regenerate=True)
        assert {c[0] for c in calls} == {"tex", "png", "pdf", "svg"}


# ---------------------------------------------------------------------------
# Tests for generate_rst_table
# ---------------------------------------------------------------------------


@pytest.mark.catalog_update
class TestGenerateRstTable:
    """Tests for ``generate_rst_table(df, rst_path, ...)``, with image generation mocked."""

    def _generate(self, tmp_path, monkeypatch, *rows, regenerate=False):
        _record_calls(monkeypatch)
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir(exist_ok=True)
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(
            _make_df(*rows), rst_path, regenerate_images=regenerate, catalog_dir=catalog_dir
        )
        return rst_path.read_text()

    def test_efg_row_produces_rst_with_title_load_and_downloads(self, tmp_path, monkeypatch):
        slug = "fakeauthor2000/fig1"
        rst = self._generate(tmp_path, monkeypatch, _efg_row(slug, title="Fake Figure 1"))
        assert "Fake Figure 1" in rst
        assert f'pygambit.catalog.load("{slug}")' in rst
        assert f":download:`fig1.efg <../games/{slug}/fig1.efg>`" in rst
        assert f":download:`fig1.ef <../games/{slug}/fig1.ef>`" in rst
        assert f":download:`fig1-default.pdf <../games/img/{slug}/default.pdf>`" in rst

    def test_single_drawing_shown_as_image(self, tmp_path, monkeypatch):
        slug = "fakeauthor2000/fig1"
        rst = self._generate(tmp_path, monkeypatch, _efg_row(slug))
        assert f".. image:: ../games/img/{slug}/default.svg" in rst
        assert ".. tab-set::" not in rst

    def test_nfg_row_shown_as_image(self, tmp_path, monkeypatch):
        slug = "fakeauthor2001/matrix1"
        _write_nfg_game(tmp_path / "catalog" / slug / "matrix1.nfg")
        rst = self._generate(tmp_path, monkeypatch, _nfg_row(slug))
        assert f".. image:: ../games/img/{slug}/default.svg" in rst
        assert f":download:`matrix1.nfg <../games/{slug}/matrix1.nfg>`" in rst

    def test_several_drawings_produce_tab_set(self, tmp_path, monkeypatch):
        slug = "fakevariant2001/fig1"
        drawings = [
            {"key": "default", "label": "Default", "path": f"{slug}/fig1.ef"},
            {"key": "wide", "label": "Wide", "path": f"{slug}/fig1__wide.ef"},
        ]
        rst = self._generate(tmp_path, monkeypatch, _efg_row(slug, drawings=drawings))
        assert ".. tab-set::" in rst
        assert ".. tab-item:: Default" in rst
        assert ".. tab-item:: Wide" in rst
        assert f".. image:: ../games/img/{slug}/wide.svg" in rst

    def test_unknown_format_row_is_skipped(self, tmp_path, monkeypatch):
        row = {**_efg_row("fakegame/v1", title="Fake Game"), "Format": "efg_2"}
        assert "Fake Game" not in self._generate(tmp_path, monkeypatch, row)

    def test_row_without_description_is_skipped(self, tmp_path, monkeypatch):
        row = _efg_row("fakeauthor2000/fig1", description="")
        assert "fakeauthor2000/fig1" not in self._generate(tmp_path, monkeypatch, row)


# ---------------------------------------------------------------------------
# Tests for grouping by work
# ---------------------------------------------------------------------------


def _source(key, citation, title="A Work"):
    return {**_FAKE_SOURCE, "key": key, "citation": citation, "title": title}


@pytest.mark.catalog_update
class TestGroupByWork:
    """Tests for ``_group_by_work`` and the RST ``generate_rst_table`` writes from it."""

    def test_games_grouped_by_source_key(self):
        a = _source("A", "Alpha (2001)")
        df = _make_df(_efg_row("x/g2", source=a), _efg_row("y/g1", source=a), _efg_row("z/g3"))
        works = build._group_by_work(df)
        assert [(w["key"], [r["Game"] for r in rows]) for w, rows in works] == [
            ("A", ["x/g2", "y/g1"]),
            ("Fake2000", ["z/g3"]),
        ]

    def test_works_sorted_by_citation_and_games_by_slug(self):
        df = _make_df(
            _efg_row("b/g", source=_source("K1", "Zeta (1990)")),
            _efg_row("a/h", source=_source("K2", "Alpha (2001)")),
            _efg_row("a/g", source=_source("K2", "Alpha (2001)")),
        )
        works = build._group_by_work(df)
        assert [w["citation"] for w, _ in works] == ["Alpha (2001)", "Zeta (1990)"]
        assert [r["Game"] for r in works[0][1]] == ["a/g", "a/h"]

    def test_skips_unknown_format_and_empty_description(self):
        df = _make_df(
            {**_efg_row("a/g"), "Format": "xyz"}, _efg_row("a/h", description=""), _efg_row("a/i")
        )
        ((_, rows),) = build._group_by_work(df)
        assert [r["Game"] for r in rows] == ["a/i"]

    def _rst(self, tmp_path, monkeypatch, *rows):
        _record_calls(monkeypatch)
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(_make_df(*rows), rst_path, catalog_dir=tmp_path / "catalog")
        return rst_path.read_text()

    def test_work_heading_is_citation_and_title(self, tmp_path, monkeypatch):
        rst = self._rst(
            tmp_path, monkeypatch, _efg_row("a/g", source=_source("K", "Alpha (2001)", "On Games"))
        )
        assert rst.startswith(".. dropdown:: Alpha (2001): On Games\n")

    def test_one_work_dropdown_holds_all_its_games(self, tmp_path, monkeypatch):
        rst = self._rst(tmp_path, monkeypatch, _efg_row("a/g"), _efg_row("a/h"))
        assert rst.count(".. dropdown:: Fake (2000): A Fake Work") == 1
        assert "   .. dropdown:: " in rst

    def test_game_heading_is_locator_and_name(self, tmp_path, monkeypatch):
        source = {**_FAKE_SOURCE, "locator": "Figure 2"}
        rst = self._rst(tmp_path, monkeypatch, _efg_row("a/g", source=source, name="The horse"))
        assert "   .. dropdown:: Figure 2: The horse\n      :open:" in rst

    def test_game_heading_falls_back_to_name_then_title(self, tmp_path, monkeypatch):
        rst = self._rst(tmp_path, monkeypatch, _efg_row("a/g", name="Only name"))
        assert "   .. dropdown:: Only name\n" in rst
        rst = self._rst(tmp_path, monkeypatch, _efg_row("a/g", title="Full title"))
        assert "   .. dropdown:: Full title\n" in rst

    def test_no_list_table_in_output(self, tmp_path, monkeypatch):
        rst = self._rst(tmp_path, monkeypatch, _efg_row("a/g"))
        assert ".. list-table::" not in rst


@pytest.mark.catalog_update
class TestWarnMissingDescriptions:
    """Tests for ``_warn_missing_descriptions(df)``.

    The function prints to stderr for each game in a supported format that has
    an empty description.  Tests use ``capsys`` to capture stderr output.
    """

    def test_game_without_description_warns(self, capsys):
        """A game with an empty description produces a WARNING on stderr."""
        df = _make_df(_efg_row("journals/nobody2025/fig1", description=""))
        build._warn_missing_descriptions(df)
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "journals/nobody2025/fig1" in err

    def test_game_with_description_does_not_warn(self, capsys):
        """A game with a non-empty description produces no output."""
        df = _make_df(_efg_row("journals/nobody2025/fig1"))
        build._warn_missing_descriptions(df)
        err = capsys.readouterr().err
        assert err == ""


# ---------------------------------------------------------------------------
# Tests for the GUI-facing manifest (citation resolution, stats, build/write)
# ---------------------------------------------------------------------------


@pytest.mark.catalog_update
class TestCitationResolution:
    """Tests for ``load_citation_texts`` and ``resolve_citations``."""

    def test_single_author(self, tmp_path):
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        assert texts["Solo2020"] == "(Solo 2020)"

    def test_two_authors(self, tmp_path):
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        assert texts["Duo2019"] == "(First and Second 2019)"

    def test_three_or_more_authors_uses_et_al(self, tmp_path):
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        assert texts["Trio2018"] == "(One et al. 2018)"

    def test_von_particle_kept_with_last_name(self, tmp_path):
        """A "von"-style name particle must not be dropped (gambit's own bibliography
        is full of these, e.g. "von Stengel")."""
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        assert texts["VonParticle2021"] == "(von Particle 2021)"

    def test_resolve_citations_substitutes_known_keys(self, tmp_path):
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        result = build.resolve_citations("See :cite:p:`Solo2020` for details.", texts)
        assert result == "See (Solo 2020) for details."

    def test_resolve_citations_substitutes_multiple_keys(self, tmp_path):
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        result = build.resolve_citations(
            "Compare :cite:p:`Solo2020` and :cite:p:`Duo2019`.", texts
        )
        assert result == "Compare (Solo 2020) and (First and Second 2019)."

    def test_resolve_citations_rejects_unknown_keys(self, tmp_path):
        texts = build.load_citation_texts(_write_bib(tmp_path / "refs.bib"))
        with pytest.raises(build.CatalogError, match="'Missing2099', 'Gone1900'"):
            build.resolve_citations(
                ":cite:p:`Missing2099`, :cite:p:`Solo2020` and :cite:p:`Gone1900`.", texts
            )

    def test_cited_keys_in_order(self):
        assert build.cited_keys(":cite:p:`B` then :cite:p:`A`; no key here.") == ["B", "A"]

    def test_resolve_citations_no_markup_is_unchanged(self):
        assert build.resolve_citations("Plain text, no citations.", {}) == (
            "Plain text, no citations."
        )


def _work(tmp_path, key):
    refs = build.load_references(_write_bib(tmp_path / "refs.bib"))
    return build.describe_work(refs[key])


@pytest.mark.catalog_update
class TestDescribeWork:
    """Tests for ``describe_work``: the manifest's description of a bibliography entry."""

    def test_one_author_article(self, tmp_path):
        assert _work(tmp_path, "Solo2020") == {
            "citation": "Solo (2020)",
            "authors": ["A. Solo"],
            "year": 2020,
            "title": "A Solo Paper",
            "kind": "article",
            "venue": "Journal of Tests",
        }

    def test_two_authors_joined_with_and(self, tmp_path):
        work = _work(tmp_path, "Duo2019")
        assert work["citation"] == "First and Second (2019)"
        assert work["authors"] == ["A. First", "B. Second"]

    def test_three_authors_use_et_al_and_markup_is_stripped(self, tmp_path):
        work = _work(tmp_path, "Conf2016")
        assert work["citation"] == "S\u00f8rensen et al. (2016)"
        assert work["authors"][0] == "T. B. S\u00f8rensen"
        assert len(work["authors"]) == 3
        assert work["title"] == "Proceedings Paper: \u00dcber Games"

    def test_inproceedings_venue_is_booktitle(self, tmp_path):
        work = _work(tmp_path, "Conf2016")
        assert work["kind"] == "inproceedings"
        assert work["venue"] == "Proceedings of the Test Conference"

    def test_book_has_no_venue(self, tmp_path):
        work = _work(tmp_path, "Book2010")
        assert work["kind"] == "book"
        assert "venue" not in work

    def test_particle_kept_in_citation_and_name(self, tmp_path):
        work = _work(tmp_path, "VonParticle2021")
        assert work["citation"] == "von Particle (2021)"
        assert work["authors"] == ["X. Y. von Particle"]

    def test_editors_stand_in_when_there_are_no_authors(self, tmp_path):
        work = _work(tmp_path, "Edited2005")
        assert work["citation"] == "Editor (2005)"
        assert work["authors"] == ["E. Editor"]

    def test_missing_year_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="year"):
            _work(tmp_path, "NoYear")


@pytest.mark.catalog_update
class TestSourceFile:
    """Tests for ``read_source_file`` and ``resolve_source``."""

    def _read(self, tmp_path, text):
        path = tmp_path / "source.yaml"
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return build.read_source_file(path)

    def test_full_file(self, tmp_path):
        data = self._read(
            tmp_path,
            """\
            source:
              key: Sel75
              locator: Figure 2
            origin:
              key: Lie60
              locator: Table 1
              note: payoffs rescaled
            name: Selten's horse
            """,
        )
        assert data == {
            "source": {"key": "Sel75", "locator": "Figure 2"},
            "origin": {"key": "Lie60", "locator": "Table 1", "note": "payoffs rescaled"},
            "name": "Selten's horse",
        }

    def test_minimal_file(self, tmp_path):
        assert self._read(tmp_path, "source:\n  key: Sel75\n") == {"source": {"key": "Sel75"}}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("name: x\n", "needs a 'source'"),
            ("source:\n  locator: Figure 1\n", "'source' needs a 'key'"),
            ("source:\n  key: A\n  page: 3\n", "unknown key 'page' in 'source'"),
            ("source:\n  key: A\nauthor: B\n", "unknown key 'author'"),
            ("source:\n  key: A\norigin:\n  locator: T\n", "'origin' needs a 'key'"),
            (
                "source:\n  key: A\norigin:\n  key: B\n  year: 1\n",
                "unknown key 'year' in 'origin'",
            ),
            ("source:\n  key: A\n  locator: 2\n", "'source.locator' must be a non-empty string"),
            ("source: A\n", "'source' must be a mapping"),
            ("source:\n  key: A\nname: [x]\n", "'name' must be a non-empty string"),
            ("- source\n", "must be a mapping"),
            ("source: [unclosed\n", "not valid YAML"),
        ],
    )
    def test_invalid_file_rejected_naming_the_file(self, tmp_path, text, message):
        with pytest.raises(build.CatalogError, match=message) as excinfo:
            self._read(tmp_path, text)
        assert str(tmp_path / "source.yaml") in str(excinfo.value)

    def test_resolve_with_origin(self, tmp_path):
        refs = build.load_references(_write_bib(tmp_path / "refs.bib"))
        path = _write_source(
            tmp_path / "g",
            "Solo2020",
            "Figure 2",
            "Horse",
            origin={"key": "Duo2019", "locator": "Table 1", "note": "rescaled"},
        )
        resolved = build.resolve_source(path, refs)
        assert list(resolved["source"])[:3] == ["key", "locator", "citation"]
        assert resolved["source"]["citation"] == "Solo (2020)"
        assert resolved["origin"]["citation"] == "First and Second (2019)"
        assert resolved["origin"]["note"] == "rescaled"
        assert resolved["name"] == "Horse"

    def test_resolve_without_origin_or_name(self, tmp_path):
        refs = build.load_references(_write_bib(tmp_path / "refs.bib"))
        resolved = build.resolve_source(_write_source(tmp_path / "g"), refs)
        assert set(resolved) == {"source"}
        assert "locator" not in resolved["source"]

    def test_unknown_key_rejected(self, tmp_path):
        refs = build.load_references(_write_bib(tmp_path / "refs.bib"))
        path = _write_source(tmp_path / "g", "Nope1999")
        with pytest.raises(build.CatalogError, match="Nope1999.*not in references.bib"):
            build.resolve_source(path, refs)

    def test_unknown_origin_key_rejected(self, tmp_path):
        refs = build.load_references(_write_bib(tmp_path / "refs.bib"))
        path = _write_source(tmp_path / "g", origin={"key": "Nope1999"})
        with pytest.raises(build.CatalogError, match="origin.key.*Nope1999"):
            build.resolve_source(path, refs)

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(build.CatalogError, match="source.yaml: missing"):
            build.resolve_source(tmp_path / "g" / "source.yaml", {})


@pytest.mark.catalog_update
class TestDerivedTitle:
    """Tests for ``derived_title``."""

    def _title(self, tmp_path, **kwargs):
        refs = build.load_references(_write_bib(tmp_path / "refs.bib"))
        return build.derived_title(
            build.resolve_source(_write_source(tmp_path / "g", **kwargs), refs)
        )

    def test_locator_only(self, tmp_path):
        assert self._title(tmp_path, locator="Figure 2") == "Solo (2020), Figure 2"

    def test_locator_and_name(self, tmp_path):
        title = self._title(tmp_path, locator="Figure 1", name="Selten's horse")
        assert title == "Solo (2020), Figure 1: Selten's horse"

    def test_citation_only(self, tmp_path):
        assert self._title(tmp_path) == "Solo (2020)"

    def test_name_without_locator(self, tmp_path):
        assert self._title(tmp_path, name="Centipede") == "Solo (2020): Centipede"

    def test_origin_does_not_enter_the_title(self, tmp_path):
        title = self._title(tmp_path, locator="Table 1", origin={"key": "Duo2019"})
        assert title == "Solo (2020), Table 1"


@pytest.mark.catalog_update
class TestTitleSync:
    """Tests for ``catalog_problems`` and ``sync_titles``."""

    def _catalog(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(catalog_dir, "src/ok", locator="Figure 1")
        _add_game(catalog_dir, "src/bad", locator="Figure 2", title="Something else")
        _add_game(catalog_dir, "src/nfg", key="Duo2019", name="Dilemma", fmt="nfg", title="Old")
        return catalog_dir, _write_bib(tmp_path / "refs.bib")

    def test_problems_list_every_mismatch(self, tmp_path):
        catalog_dir, bib = self._catalog(tmp_path)
        problems = build.catalog_problems(catalog_dir, bib)
        assert len(problems) == 2
        assert "src/bad" in problems[0]
        assert "'Something else'" in problems[0]
        assert "'Solo (2020), Figure 2'" in problems[0]
        assert "src/nfg" in problems[1]

    def test_problems_list_every_missing_description_key(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(catalog_dir, "src/one", description="See :cite:p:`Missing2099`.")
        _add_game(
            catalog_dir,
            "src/two",
            description=":cite:p:`Duo2019`, :cite:p:`Gone1900` and :cite:p:`Lost1800`.",
        )
        problems = build.catalog_problems(catalog_dir, _write_bib(tmp_path / "refs.bib"))
        assert problems == [
            "src/one: description cites 'Missing2099', not in references.bib",
            "src/two: description cites 'Gone1900', not in references.bib",
            "src/two: description cites 'Lost1800', not in references.bib",
        ]

    def test_build_fails_on_missing_description_key(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(catalog_dir, "src/one", description="See :cite:p:`Missing2099`.")
        with pytest.raises(build.CatalogError, match="Missing2099"):
            build.build_manifest(catalog_dir, _write_bib(tmp_path / "refs.bib"), "0.1.0")

    def test_no_problems_when_titles_derived(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(catalog_dir, "src/ok", locator="Figure 1")
        assert build.catalog_problems(catalog_dir, _write_bib(tmp_path / "refs.bib")) == []

    def test_sync_rewrites_only_the_title(self, tmp_path):
        catalog_dir, bib = self._catalog(tmp_path)
        efg = catalog_dir / "src/bad/bad.efg"
        nfg = catalog_dir / "src/nfg/nfg.nfg"
        efg_before, nfg_before = efg.read_bytes(), nfg.read_bytes()
        changed = build.sync_titles(catalog_dir, bib)
        assert changed == ["src/bad", "src/nfg"]
        assert efg.read_bytes() == efg_before.replace(b"Something else", b"Solo (2020), Figure 2")
        assert nfg.read_bytes() == nfg_before.replace(
            b'"Old"', b'"First and Second (2019): Dilemma"'
        )
        assert build.catalog_problems(catalog_dir, bib) == []

    def test_sync_leaves_correct_files_untouched(self, tmp_path):
        catalog_dir, bib = self._catalog(tmp_path)
        ok = catalog_dir / "src/ok/ok.efg"
        before = ok.stat().st_mtime_ns
        build.sync_titles(catalog_dir, bib)
        assert ok.stat().st_mtime_ns == before

    def test_sync_preserves_line_endings(self, tmp_path):
        catalog_dir, bib = self._catalog(tmp_path)
        efg = catalog_dir / "src/bad/bad.efg"
        efg.write_bytes(efg.read_bytes().replace(b"\n", b"\r\n"))
        build.sync_titles(catalog_dir, bib)
        data = efg.read_bytes()
        assert data.startswith(b'EFG 2 R "Solo (2020), Figure 2" { "P1" "P2" }\r\n')
        assert data.count(b"\n") == data.count(b"\r\n")

    def test_check_mode_writes_nothing(self, tmp_path):
        catalog_dir, bib = self._catalog(tmp_path)
        efg = catalog_dir / "src/bad/bad.efg"
        before = efg.read_bytes()
        assert build.sync_titles(catalog_dir, bib, write=False) == ["src/bad", "src/nfg"]
        assert efg.read_bytes() == before

    def test_title_with_quote_is_escaped(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(catalog_dir, "src/g", name='The "game"', title="Old")
        build.sync_titles(catalog_dir, _write_bib(tmp_path / "refs.bib"))
        game = build.read_game(build.discover_entries(catalog_dir)[0])
        assert game.title == 'Solo (2020): The "game"'

    def test_sync_refuses_when_a_source_is_missing(self, tmp_path):
        catalog_dir, bib = self._catalog(tmp_path)
        (catalog_dir / "src/ok/source.yaml").unlink()
        with pytest.raises(build.CatalogError, match="src/ok/source.yaml: missing"):
            build.sync_titles(catalog_dir, bib)


@pytest.mark.catalog_update
class TestGameStats:
    """Tests for ``_game_stats``."""

    def test_stats_of_extensive_form_game(self, tmp_path):
        """Player 1 has 2 strategies (one decision, two actions); player 2 has no
        decision nodes at all, so the reduced normal form gives them a single
        trivial strategy — 2 + 1 = 3 total, not 2x2=4."""
        efg_path = _write_efg_game(tmp_path / "game1.efg")
        game = build.gbt.read_efg(str(efg_path))
        stats = build._game_stats(game)
        assert stats == {
            "n_players": 2,
            "is_const_sum": True,
            "n_strategies": 3,
        }

    def test_stats_of_strategic_form_game(self, tmp_path):
        nfg_path = _write_nfg_game(tmp_path / "game1.nfg")
        game = build.gbt.read_nfg(str(nfg_path))
        stats = build._game_stats(game)
        assert stats["n_players"] == 2
        assert stats["n_strategies"] == 4


@pytest.mark.catalog_update
class TestBuildManifest:
    """Tests for ``build_manifest`` and ``write_manifest``."""

    def _make_catalog(self, tmp_path):
        """Build a small fake catalog dir with one EFG and one NFG game."""
        catalog_dir = tmp_path / "catalog"
        _add_game(
            catalog_dir,
            "a/game1",
            locator="Figure 1",
            name="Tree game",
            origin={"key": "Book2010", "locator": "Table 3", "note": "rescaled"},
            description="A tree game from :cite:p:`Solo2020`.",
        )
        _add_game(
            catalog_dir,
            "b/game2",
            key="Duo2019",
            fmt="nfg",
            description="A table game from :cite:p:`Duo2019`.",
        )
        return catalog_dir

    def _build(self, catalog_dir, tmp_path):
        return build.build_manifest(
            catalog_dir=catalog_dir, bib_path=_write_bib(tmp_path / "refs.bib"), version="2.3.4"
        )

    def test_build_manifest_versions(self, tmp_path):
        manifest = self._build(self._make_catalog(tmp_path), tmp_path)
        assert manifest["schema_version"] == build.SCHEMA_VERSION == 1
        assert manifest["catalog_version"] == "2.3.4"

    def test_build_manifest_entries(self, tmp_path):
        catalog_dir = self._make_catalog(tmp_path)
        entries = self._build(catalog_dir, tmp_path)["games"]
        by_slug = {e["slug"]: e for e in entries}
        assert set(by_slug) == {"a/game1", "b/game2"}

        efg_entry = by_slug["a/game1"]
        assert efg_entry["title"] == "Solo (2020), Figure 1: Tree game"
        assert efg_entry["name"] == "Tree game"
        assert efg_entry["description"] == "A tree game from (Solo 2020)."
        assert efg_entry["format"] == "efg"
        assert efg_entry["game"] == "a/game1/game1.efg"
        assert efg_entry["drawings"] == []
        assert "category" not in efg_entry
        assert "group" not in efg_entry

        nfg_entry = by_slug["b/game2"]
        assert nfg_entry["format"] == "nfg"
        assert nfg_entry["description"] == "A table game from (First and Second 2019)."

    def test_source_and_origin(self, tmp_path):
        by_slug = {
            e["slug"]: e for e in self._build(self._make_catalog(tmp_path), tmp_path)["games"]
        }
        entry = by_slug["a/game1"]
        assert entry["source"] == {
            "key": "Solo2020",
            "locator": "Figure 1",
            "citation": "Solo (2020)",
            "authors": ["A. Solo"],
            "year": 2020,
            "title": "A Solo Paper",
            "venue": "Journal of Tests",
            "kind": "article",
        }
        assert entry["origin"] == {
            "key": "Book2010",
            "locator": "Table 3",
            "citation": "Writer (2010)",
            "authors": ["W. Writer"],
            "year": 2010,
            "title": "The Book",
            "kind": "book",
            "note": "rescaled",
        }

    def test_origin_locator_and_name_omitted_when_absent(self, tmp_path):
        entry = self._build(self._make_catalog(tmp_path), tmp_path)["games"][1]
        assert entry["slug"] == "b/game2"
        assert "origin" not in entry
        assert "name" not in entry
        assert "locator" not in entry["source"]

    def test_build_manifest_skips_games_without_description(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _add_game(catalog_dir, "nodesc/game1", description="")
        assert self._build(catalog_dir, tmp_path)["games"] == []

    def test_build_manifest_sorted_by_slug(self, tmp_path):
        entries = self._build(self._make_catalog(tmp_path), tmp_path)["games"]
        assert [e["slug"] for e in entries] == sorted(e["slug"] for e in entries)

    def test_built_manifest_validates(self, tmp_path):
        build.validate_manifest(self._build(self._make_catalog(tmp_path), tmp_path))

    def test_build_fails_listing_every_missing_source(self, tmp_path):
        catalog_dir = self._make_catalog(tmp_path)
        (catalog_dir / "a/game1/source.yaml").unlink()
        (catalog_dir / "b/game2/source.yaml").unlink()
        with pytest.raises(build.CatalogError) as excinfo:
            self._build(catalog_dir, tmp_path)
        message = str(excinfo.value)
        assert "a/game1/source.yaml: missing" in message
        assert "b/game2/source.yaml: missing" in message

    def test_build_fails_on_unknown_key(self, tmp_path):
        catalog_dir = self._make_catalog(tmp_path)
        _write_source(catalog_dir / "a/game1", "Nope1999")
        with pytest.raises(build.CatalogError, match="Nope1999"):
            self._build(catalog_dir, tmp_path)

    def test_build_fails_on_title_mismatch(self, tmp_path):
        catalog_dir = self._make_catalog(tmp_path)
        _write_efg_game(catalog_dir / "a/game1/game1.efg", title="Hand-written", description="D")
        with pytest.raises(build.CatalogError, match="a/game1: title is 'Hand-written'"):
            self._build(catalog_dir, tmp_path)

    def test_build_fails_on_invalid_source_file(self, tmp_path):
        catalog_dir = self._make_catalog(tmp_path)
        _write_source(catalog_dir / "a/game1", extra="colour: red\n")
        with pytest.raises(build.CatalogError, match="unknown key 'colour'"):
            self._build(catalog_dir, tmp_path)

    def test_write_manifest_creates_file(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        build.write_manifest([{"slug": "a/b"}], manifest_path=manifest_path)
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == [{"slug": "a/b"}]

    def test_write_manifest_no_op_when_unchanged(self, tmp_path, capsys):
        manifest_path = tmp_path / "manifest.json"
        build.write_manifest([{"slug": "a/b"}], manifest_path=manifest_path)
        capsys.readouterr()
        build.write_manifest([{"slug": "a/b"}], manifest_path=manifest_path)
        assert "No changes" in capsys.readouterr().out

    def test_write_manifest_updates_on_change(self, tmp_path, capsys):
        manifest_path = tmp_path / "manifest.json"
        build.write_manifest([{"slug": "a/b"}], manifest_path=manifest_path)
        capsys.readouterr()
        build.write_manifest([{"slug": "a/c"}], manifest_path=manifest_path)
        assert "Updated" in capsys.readouterr().out


@pytest.mark.catalog_update
class TestManifestSchema:
    """Tests for ``validate_manifest`` against ``schema/manifest.schema.json``."""

    def _committed(self):
        return json.loads(build.CATALOG_MANIFEST.read_text(encoding="utf-8"))

    def _manifest(self, tmp_path):
        """Return a manifest built from a small fixture catalog, with an origin and a name."""
        catalog_dir = tmp_path / "catalog"
        _add_game(
            catalog_dir,
            "a/game1",
            locator="Figure 1",
            name="Named",
            origin={"key": "Book2010"},
            description="D.",
        )
        return build.build_manifest(catalog_dir, _write_bib(tmp_path / "refs.bib"), "1.0.0")

    def test_fixture_manifest_validates(self, tmp_path):
        build.validate_manifest(self._manifest(tmp_path))

    def test_committed_manifest_validates(self):
        build.validate_manifest(self._committed())

    def test_committed_manifest_matches_version_file(self):
        manifest = self._committed()
        assert manifest["catalog_version"] == build.catalog_version()
        assert manifest["schema_version"] == build.SCHEMA_VERSION

    def test_committed_manifest_is_up_to_date(self):
        manifest = self._committed()
        assert manifest == build.build_manifest(version=manifest["catalog_version"])

    def test_missing_required_field_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        del manifest["games"][0]["format"]
        with pytest.raises(jsonschema.ValidationError, match="format"):
            build.validate_manifest(manifest)

    def test_unknown_field_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["thumbnail"] = "img/x.png"
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_dropped_fields_fail(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["category"] = "Books"
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_missing_source_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        del manifest["games"][0]["source"]
        with pytest.raises(jsonschema.ValidationError, match="source"):
            build.validate_manifest(manifest)

    def test_unknown_source_field_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["source"]["page"] = 3
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_empty_description_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["description"] = ""
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_empty_title_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["title"] = ""
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    @pytest.mark.parametrize("version", [0, 2, "1"])
    def test_other_schema_version_fails(self, tmp_path, version):
        manifest = self._manifest(tmp_path)
        manifest["schema_version"] = version
        with pytest.raises(jsonschema.ValidationError, match="schema_version|1 was expected"):
            build.validate_manifest(manifest)

    def test_note_on_source_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["source"]["note"] = "payoffs rescaled"
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_note_on_origin_validates(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["origin"]["note"] = "payoffs rescaled"
        build.validate_manifest(manifest)

    def test_unknown_origin_field_fails(self, tmp_path):
        manifest = self._manifest(tmp_path)
        manifest["games"][0]["origin"]["page"] = 3
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_every_extensive_game_has_a_primary_drawing(self):
        missing = [
            g["slug"]
            for g in self._committed()["games"]
            if g["format"] == "efg" and (not g["drawings"] or g["drawings"][0]["key"] != "default")
        ]
        assert missing == []


# ---------------------------------------------------------------------------
# Tests for the release archive
# ---------------------------------------------------------------------------


@pytest.mark.catalog_update
class TestBuildRelease:
    """Tests for ``build_release``, on a small fake repository with stubbed image generation."""

    def _make_repo(self, tmp_path, monkeypatch, version="0.9.1"):
        root = tmp_path / "repo"
        catalog_dir = root / "games"
        _add_game(catalog_dir, "src2000/fig1", description="A tree.")
        (catalog_dir / "src2000" / "fig1" / "fig1.ef").write_text("level 0 node 1\n")
        _add_game(catalog_dir, "src2000/tab1", fmt="nfg", description="A table.")
        for name in ["LICENSE", "README.md", "LICENSES/CC-BY-4.0.txt"]:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(name)
        (root / "schema").mkdir()
        (root / "schema" / "manifest.schema.json").write_bytes(build.MANIFEST_SCHEMA.read_bytes())
        bib = _write_bib(tmp_path / "refs.bib")
        monkeypatch.setattr(build, "REFERENCES_BIB", bib)
        manifest = build.build_manifest(catalog_dir, bib_path=bib, version=version)
        build.write_manifest(manifest, catalog_dir / "manifest.json")
        monkeypatch.setattr(build, "png", _fake_png)
        return catalog_dir

    def test_archive_contents(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)
        archive = build.build_release(tmp_path / "out", catalog_dir=catalog_dir)
        assert archive.name == "gambit-catalog-0.9.1.zip"
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        top = "gambit-catalog-0.9.1/"
        assert names == {
            top + n
            for n in [
                "games/manifest.json",
                "games/src2000/fig1/fig1.efg",
                "games/src2000/fig1/fig1.ef",
                "games/src2000/tab1/tab1.nfg",
                "thumbs/src2000/fig1.png",
                "thumbs/src2000/tab1.png",
                "LICENSE",
                "LICENSES/CC-BY-4.0.txt",
                "README.md",
                "schema/manifest.schema.json",
            ]
        }

    def test_thumbnail_size(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)
        archive = build.build_release(tmp_path / "out", catalog_dir=catalog_dir)
        from PIL import Image

        with zipfile.ZipFile(archive) as zf:
            data = zf.read("gambit-catalog-0.9.1/thumbs/src2000/fig1.png")
        with Image.open(io.BytesIO(data)) as im:
            assert max(im.size) == 560

    def test_sha256_file_matches(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)
        archive = build.build_release(tmp_path / "out", catalog_dir=catalog_dir)
        line = (archive.parent / f"{archive.name}.sha256").read_text()
        assert line == f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"

    def test_archive_is_reproducible(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)
        a = build.build_release(tmp_path / "out1", catalog_dir=catalog_dir).read_bytes()
        b = build.build_release(tmp_path / "out2", catalog_dir=catalog_dir).read_bytes()
        assert a == b

    def test_missing_thumbnail_fails_naming_slug(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)

        def failing_png(source, save_to, style):
            if "tab1" in save_to:
                raise RuntimeError("pdflatex failed")
            _fake_png(source, save_to, style)

        monkeypatch.setattr(build, "png", failing_png)
        with pytest.raises(RuntimeError, match="src2000/tab1"):
            build.build_release(tmp_path / "out", catalog_dir=catalog_dir)

    def test_extensive_game_without_drawing_fails(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)
        (catalog_dir / "src2000" / "fig1" / "fig1.ef").unlink()
        manifest = build.build_manifest(catalog_dir, version="0.9.1")
        build.write_manifest(manifest, catalog_dir / "manifest.json")
        with pytest.raises(RuntimeError, match="src2000/fig1"):
            build.build_release(tmp_path / "out", catalog_dir=catalog_dir)

    def test_stale_manifest_fails(self, tmp_path, monkeypatch):
        catalog_dir = self._make_repo(tmp_path, monkeypatch)
        _add_game(catalog_dir, "src2000/new", description="New.")
        with pytest.raises(RuntimeError, match="out of date"):
            build.build_release(tmp_path / "out", catalog_dir=catalog_dir)
