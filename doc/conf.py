import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import _bibstyle  # noqa: E402,F401  registers the "keystyle" pybtex plugin

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = [str(pathlib.Path(__file__).parent.parent / "references.bib")]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"

project = "Gambit Catalog"
copyright = "1994-2026, The Gambit Project"  # noqa

release = (pathlib.Path(__file__).parent.parent / "VERSION").read_text().strip()
version = ".".join(release.split(".")[:2])

exclude_patterns = ["_build", "_table.rst"]

pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "external_links": [
        {"name": "Gambit documentation", "url": "https://gambitproject.readthedocs.io/"},
        {"name": "GitHub", "url": "https://github.com/gambitproject/gambit-catalog"},
    ],
}
