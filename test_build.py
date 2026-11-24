"""Tests for build.py.

All catalog slugs used here are clearly fictional (e.g. ``"testgroup2000/fig1"``)
and do not correspond to any game in the real catalog.  This is intentional: the
tests construct their own temporary catalog directories and DataFrame rows so they
are completely isolated from the actual catalog on disk.

Monkeypatching strategy
-----------------------
``build.py`` depends on three external resources that are replaced in tests:

1. ``CATALOG_HIERARCHY_CONFIG`` (a ``Path``) — swapped for a tmp YAML file so
   ``load_hierarchy_labels`` reads controlled labels without touching the real
   ``hierarchy.yaml``.  Swap via ``monkeypatch.setattr(build,
   "CATALOG_HIERARCHY_CONFIG", yaml_file)``.

2. ``tex`` / ``png`` / ``pdf`` / ``svg``
   (functions imported from ``gtdraw``) — replaced with no-ops, call-tracking
   lambdas, or a stub that writes a small PNG.  This lets us test RST and release
   generation without invoking LaTeX, and assert whether image generation was
   triggered at all.

3. ``catalog_dir`` (an argument to ``generate_rst_table`` and ``build_release``) —
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


def _efg_row(slug, title="Test EFG Game", description="A description.", drawings=None):
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
    }


def _nfg_row(slug, title="Test NFG Game", description="A description."):
    """Return a dict representing one row of the DataFrame for a normal-form game."""
    stem = slug.rsplit("/", 1)[-1]
    return {
        "Game": slug,
        "Title": title,
        "Description": description,
        "Format": "nfg",
        "Path": f"{slug}/{stem}.nfg",
        "Drawings": [],
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

    def test_catalog_games_reads_title_and_description(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(
            catalog_dir / "src2000" / "fig1" / "fig1.efg", title="T", description="D"
        )
        df = build._catalog_games(catalog_dir)
        assert df.to_dict("records") == [
            {
                "Game": "src2000/fig1",
                "Title": "T",
                "Description": "D",
                "Format": "efg",
                "Path": "src2000/fig1/fig1.efg",
                "Drawings": [],
            }
        ]


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
# Tests for hierarchy helpers and hierarchical RST output
# ---------------------------------------------------------------------------

# A minimal hierarchy.yaml used by hierarchy tests.
_HIERARCHY_YAML = textwrap.dedent("""\
    labels:
      cat: "My Category"
      cat/src: "My Source"
""")


@pytest.mark.catalog_update
class TestHierarchyHelpers:
    """Unit tests for ``load_hierarchy_labels``, ``_node_label``, and ``_build_slug_tree``."""

    def test_load_hierarchy_labels_returns_dict(self, tmp_path, monkeypatch):
        """``load_hierarchy_labels`` returns the labels dict from the YAML."""
        yaml_file = tmp_path / "hier.yaml"
        yaml_file.write_text(_HIERARCHY_YAML, encoding="utf-8")
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", yaml_file)
        labels = build.load_hierarchy_labels()
        assert labels["cat"] == "My Category"
        assert labels["cat/src"] == "My Source"

    def test_node_label_uses_yaml(self, tmp_path, monkeypatch):
        """``_node_label`` returns the YAML label when the prefix is present."""
        yaml_file = tmp_path / "hier.yaml"
        yaml_file.write_text(_HIERARCHY_YAML, encoding="utf-8")
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", yaml_file)
        labels = build.load_hierarchy_labels()
        assert build._node_label("cat", labels) == "My Category"

    def test_node_label_fallback_title_case(self, tmp_path, monkeypatch):
        """``_node_label`` falls back to title-casing the last component."""
        yaml_file = tmp_path / "hier.yaml"
        yaml_file.write_text(_HIERARCHY_YAML, encoding="utf-8")
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", yaml_file)
        labels = build.load_hierarchy_labels()
        assert build._node_label("cat/unknownsrc", labels) == "Unknownsrc"

    def test_build_slug_tree_single_game(self):
        """A single-slug DataFrame builds a 2-level tree."""
        df = _make_df(_efg_row("cat/src/game1"))
        tree = build._build_slug_tree(df)
        assert "cat" in tree
        assert "src" in tree["cat"]
        assert "game1" in tree["cat"]["src"]

    def test_build_slug_tree_groups_siblings(self):
        """Two slugs sharing a prefix are grouped under the same intermediate node."""
        df = _make_df(_efg_row("cat/src/game1"), _efg_row("cat/src/game2"))
        tree = build._build_slug_tree(df)
        assert set(tree["cat"]["src"].keys()) == {"game1", "game2"}

    def test_build_slug_tree_skips_unknown_format(self):
        """Rows with unrecognised Format are excluded from the tree."""
        row = {**_efg_row("cat/src/game1"), "Format": "xyz"}
        df = _make_df(row)
        assert build._build_slug_tree(df) == {}

    def test_build_slug_tree_skips_empty_description(self):
        """Rows with an empty description are excluded from the tree."""
        df = _make_df(_efg_row("cat/src/game1", description=""))
        assert build._build_slug_tree(df) == {}


@pytest.mark.catalog_update
class TestHierarchicalRstOutput:
    """Tests that ``generate_rst_table`` produces correctly nested dropdown RST."""

    def _mock_generates(self, monkeypatch):
        for name in ["tex", "png", "pdf", "svg"]:
            monkeypatch.setattr(build, name, lambda *a, **k: None)

    def _write_hierarchy_yaml(self, tmp_path, content=_HIERARCHY_YAML):
        yaml_file = tmp_path / "hier.yaml"
        yaml_file.write_text(content, encoding="utf-8")
        return yaml_file

    def test_top_level_dropdown_is_open(self, tmp_path, monkeypatch):
        """Top-level category dropdowns carry ``:open:`` so the first level is visible."""
        self._mock_generates(monkeypatch)
        hier_yaml = self._write_hierarchy_yaml(tmp_path)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", hier_yaml)
        catalog_dir = tmp_path / "catalog"
        slug = "cat/src/game1"
        _make_image_files(catalog_dir, slug)
        df = _make_df(_efg_row(slug))
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(df, rst_path, catalog_dir=catalog_dir)
        rst = rst_path.read_text()
        assert ".. dropdown:: My Category\n   :open:" in rst

    def test_second_level_dropdown_is_not_open(self, tmp_path, monkeypatch):
        """Sub-category dropdowns do NOT carry ``:open:`` so they are collapsed by default."""
        self._mock_generates(monkeypatch)
        hier_yaml = self._write_hierarchy_yaml(tmp_path)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", hier_yaml)
        catalog_dir = tmp_path / "catalog"
        slug = "cat/src/game1"
        _make_image_files(catalog_dir, slug)
        df = _make_df(_efg_row(slug))
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(df, rst_path, catalog_dir=catalog_dir)
        rst = rst_path.read_text()
        assert "   .. dropdown:: My Source\n   \n" in rst
        # Confirm :open: does not immediately follow the second-level dropdown
        src_idx = rst.index("   .. dropdown:: My Source")
        assert ":open:" not in rst[src_idx : src_idx + 40]

    def test_game_dropdown_is_open(self, tmp_path, monkeypatch):
        """Individual game dropdowns carry ``:open:`` so game content is visible on expand."""
        self._mock_generates(monkeypatch)
        hier_yaml = self._write_hierarchy_yaml(tmp_path)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", hier_yaml)
        catalog_dir = tmp_path / "catalog"
        slug = "cat/src/game1"
        _make_image_files(catalog_dir, slug)
        df = _make_df(_efg_row(slug, title="My Game Title"))
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(df, rst_path, catalog_dir=catalog_dir)
        rst = rst_path.read_text()
        assert "      .. dropdown:: My Game Title\n         :open:" in rst

    def test_sibling_games_both_appear_under_source(self, tmp_path, monkeypatch):
        """Two games sharing a source prefix both appear nested under the source dropdown."""
        self._mock_generates(monkeypatch)
        hier_yaml = self._write_hierarchy_yaml(tmp_path)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", hier_yaml)
        catalog_dir = tmp_path / "catalog"
        for slug in ["cat/src/game1", "cat/src/game2"]:
            _make_image_files(catalog_dir, slug)
        df = _make_df(
            _efg_row("cat/src/game1", title="Game One"),
            _efg_row("cat/src/game2", title="Game Two"),
        )
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(df, rst_path, catalog_dir=catalog_dir)
        rst = rst_path.read_text()
        assert ".. dropdown:: My Category" in rst
        assert "   .. dropdown:: My Source" in rst
        assert "Game One" in rst
        assert "Game Two" in rst

    def test_no_list_table_in_output(self, tmp_path, monkeypatch):
        """The new output does not use ``.. list-table::`` (replaced by nested dropdowns)."""
        self._mock_generates(monkeypatch)
        hier_yaml = self._write_hierarchy_yaml(tmp_path)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", hier_yaml)
        catalog_dir = tmp_path / "catalog"
        slug = "cat/src/game1"
        _make_image_files(catalog_dir, slug)
        df = _make_df(_efg_row(slug))
        rst_path = tmp_path / "out.rst"
        build.generate_rst_table(df, rst_path, catalog_dir=catalog_dir)
        rst = rst_path.read_text()
        assert ".. list-table::" not in rst
        assert ".. contents::" not in rst


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

_TEST_BIB = textwrap.dedent("""\
    @article{Solo2020,
      author = {Solo, A.},
      year = {2020},
    }
    @article{Duo2019,
      author = {First, A. and Second, B.},
      year = {2019},
    }
    @article{Trio2018,
      author = {One, A. and Two, B. and Three, C.},
      year = {2018},
    }
    @article{VonParticle2021,
      author = {von Particle, X.},
      year = {2021},
    }
""")


def _write_bib(path, content=_TEST_BIB):
    path.write_text(content, encoding="utf-8")
    return path


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

    def test_resolve_citations_falls_back_for_unknown_key(self):
        result = build.resolve_citations("See :cite:p:`Missing2099`.", {})
        assert result == "See (Missing2099)."

    def test_resolve_citations_no_markup_is_unchanged(self):
        assert build.resolve_citations("Plain text, no citations.", {}) == (
            "Plain text, no citations."
        )


@pytest.mark.catalog_update
class TestHierarchyBreadcrumbs:
    """Tests for ``_hierarchy_breadcrumbs``."""

    def test_breadcrumbs_use_yaml_labels(self, tmp_path, monkeypatch):
        yaml_file = _write_yaml(tmp_path / "hier.yaml", _HIERARCHY_YAML)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", yaml_file)
        labels = build.load_hierarchy_labels()
        assert build._hierarchy_breadcrumbs("cat/src/game1", labels) == [
            "My Category",
            "My Source",
        ]

    def test_breadcrumbs_fallback_for_unknown_prefix(self, tmp_path, monkeypatch):
        yaml_file = _write_yaml(tmp_path / "hier.yaml", _HIERARCHY_YAML)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", yaml_file)
        labels = build.load_hierarchy_labels()
        assert build._hierarchy_breadcrumbs("other/unknown_group/game1", labels) == [
            "Other",
            "Unknown Group",
        ]

    def test_breadcrumbs_empty_for_top_level_slug(self, tmp_path, monkeypatch):
        """A slug with no parent directories has no breadcrumbs."""
        yaml_file = _write_yaml(tmp_path / "hier.yaml", _HIERARCHY_YAML)
        monkeypatch.setattr(build, "CATALOG_HIERARCHY_CONFIG", yaml_file)
        labels = build.load_hierarchy_labels()
        assert build._hierarchy_breadcrumbs("game1", labels) == []


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
        _write_efg_game(
            catalog_dir / "books" / "author2020" / "game1" / "game1.efg",
            title="Tree Game",
            description="A tree game from :cite:p:`Solo2020`.",
        )
        _write_nfg_game(
            catalog_dir / "journals" / "geb" / "author2019" / "game2" / "game2.nfg",
            title="Table Game",
            description="A table game from :cite:p:`Duo2019`.",
        )
        return catalog_dir

    def _build(self, catalog_dir, tmp_path):
        return build.build_manifest(
            catalog_dir=catalog_dir, bib_path=_write_bib(tmp_path / "refs.bib"), version="2.3.4"
        )

    def test_build_manifest_versions(self, tmp_path):
        manifest = self._build(self._make_catalog(tmp_path), tmp_path)
        assert manifest["schema_version"] == 2
        assert manifest["catalog_version"] == "2.3.4"

    def test_build_manifest_entries(self, tmp_path):
        catalog_dir = self._make_catalog(tmp_path)
        entries = self._build(catalog_dir, tmp_path)["games"]
        by_slug = {e["slug"]: e for e in entries}
        assert set(by_slug) == {"books/author2020/game1", "journals/geb/author2019/game2"}

        efg_entry = by_slug["books/author2020/game1"]
        assert efg_entry["title"] == "Tree Game"
        assert efg_entry["description"] == "A tree game from (Solo 2020)."
        assert efg_entry["format"] == "efg"
        assert efg_entry["game"] == "books/author2020/game1/game1.efg"
        assert efg_entry["drawings"] == []
        assert efg_entry["category"] == "Books"

        nfg_entry = by_slug["journals/geb/author2019/game2"]
        assert nfg_entry["format"] == "nfg"
        assert nfg_entry["description"] == "A table game from (First and Second 2019)."

    def test_build_manifest_skips_games_without_description(self, tmp_path):
        catalog_dir = tmp_path / "catalog"
        _write_efg_game(catalog_dir / "nodesc" / "game1" / "game1.efg", description="")
        assert self._build(catalog_dir, tmp_path)["games"] == []

    def test_build_manifest_sorted_by_slug(self, tmp_path):
        entries = self._build(self._make_catalog(tmp_path), tmp_path)["games"]
        assert [e["slug"] for e in entries] == sorted(e["slug"] for e in entries)

    def test_built_manifest_validates(self, tmp_path):
        build.validate_manifest(self._build(self._make_catalog(tmp_path), tmp_path))

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

    def test_committed_manifest_validates(self):
        build.validate_manifest(self._committed())

    def test_committed_manifest_matches_version_file(self):
        manifest = self._committed()
        assert manifest["catalog_version"] == build.catalog_version()
        assert manifest["schema_version"] == int(build.catalog_version().split(".")[0])

    def test_missing_required_field_fails(self):
        manifest = self._committed()
        del manifest["games"][0]["format"]
        with pytest.raises(jsonschema.ValidationError, match="format"):
            build.validate_manifest(manifest)

    def test_unknown_field_fails(self):
        manifest = self._committed()
        manifest["games"][0]["thumbnail"] = "img/x.png"
        with pytest.raises(jsonschema.ValidationError):
            build.validate_manifest(manifest)

    def test_empty_description_fails(self):
        manifest = self._committed()
        manifest["games"][0]["description"] = ""
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
        _write_efg_game(
            catalog_dir / "src2000" / "fig1" / "fig1.efg", title="Tree", description="A tree."
        )
        (catalog_dir / "src2000" / "fig1" / "fig1.ef").write_text("level 0 node 1\n")
        _write_nfg_game(
            catalog_dir / "src2000" / "tab1" / "tab1.nfg", title="Table", description="A table."
        )
        for name in ["LICENSE", "README.md", "LICENSES/CC-BY-4.0.txt"]:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(name)
        (root / "schema").mkdir()
        (root / "schema" / "manifest.schema.json").write_bytes(build.MANIFEST_SCHEMA.read_bytes())
        monkeypatch.setattr(build, "load_hierarchy_labels", lambda: {})
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
        _write_efg_game(catalog_dir / "src2000" / "new" / "new.efg", description="New.")
        with pytest.raises(RuntimeError, match="out of date"):
            build.build_release(tmp_path / "out", catalog_dir=catalog_dir)
