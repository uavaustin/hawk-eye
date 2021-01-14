# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "hawk-eye"
copyright = "2021, UAV Austin"
author = "UAV Austin"

docs_repo = "hawk-eye"
docs_user = "uavaustin"

releases_github_path = f"{docs_user}/{docs_repo}"


# The full version, including alpha/beta/rc tags
release = "0.0.2b0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "catalyst_sphinx_theme"
html_theme_options = {}
# html_theme_options = {
#     "display_version": True,
#     "prev_next_buttons_location": "bottom",
#     "collapse_navigation": False,
#     "sticky_navigation": True,
#     "navigation_depth": 4,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

html_context = {
    "display_github": True,
    "source_url_prefix": (
        f"https://github.com/{docs_user}/{docs_repo}/tree/master/docs"
    ),
    "github_host": "github.com",
    "github_user": docs_user,
    "github_repo": docs_repo,
    "github_version": "master",
    "conf_py_path": "/docs/",
    "source_suffix": ".rst",
}