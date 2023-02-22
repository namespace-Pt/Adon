# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# this is necessary for the local sphinx to find code from another directory
# however in the
import pathlib
import sys
# add the src folder to the code directory
sys.path.insert(0, (pathlib.Path(__file__).parents[1].resolve() / "src").as_posix())

# -- Project information -----------------------------------------------------

project = "Adon"
copyright = "2022, namespace-Pt"
author = "namespace-Pt"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    # for generating documents from docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # for google docstring
    "sphinx.ext.napoleon",
    # for markdown parsing
    "myst_parser",
    # for parsing markdown emoji
    # "sphinxemoji.sphinxemoji",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "discussions", "reviews", "backup", "ppts"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["sources/_static"]
html_css_files = ['custom.css']

# mock these packages exist
autodoc_mock_imports = ["faiss", "torch", "torch_scatter", "transformers", "pandas", "omegaconf", "hydra", "psutil", "tqdm"]

# do not change the order of classes or functions in the module
autodoc_member_order = 'bysource'

# add numbered references
numfig = True