# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))
# sys.setrecursionlimit(1000)
#
import rescomp
from rescomp._version import __version__
import sphinx.ext.graphviz

# -- Project information -----------------------------------------------------

project = 'rescomp'
copyright = '2020, Jonas Aumeier, Sebastian Baur, Joschka Herteux, Youssef Mabrouk'
author = 'Jonas Aumeier, Sebastian Baur, Joschka Herteux, Youssef Mabrouk'

# The full version, including alpha/beta/rc tags
release = __version__
version = __version__

# -- General configuration ---------------------------------------------------

# The master toctree document.
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'm2r',
    'autodocsumm',
    # 'sphinx_automodapi.automodapi',
    'sphinx.ext.graphviz',
    'sphinx.ext.autosummary',
    # 'sphinx_autopackagesummary',
    'sphinx.ext.inheritance_diagram',
    'nbsphinx',
    'nbsphinx_link',
]

autodoc_default_options = {
    'autosummary': True,
}
#
# autodoc_default_flags = ['members']
# autosummary_generate = Tru
#
# autodata_content = 'both'

# numpydoc_show_class_members = False # for automodapi

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# html_theme = "classic"
# html_theme_options = {
#     'body_min_width': 1200,
#     # Toc options
#     'stickysidebar': True,
# }


extensions.append("sphinx_rtd_theme")
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'display_version': True,
    #  Toc options
    'collapse_navigation': False,
    'includehidden': True,
}
# Override default css to get a larger page width for ReadTheDoc build. See:
# https://stackoverflow.com/questions/23211695/modifying-content-width-of-the-sphinx-theme-read-the-docs
html_css_files = [
    'theme_overrides.css',
]

# import wild_sphinx_theme
# html_theme = 'wild'
# html_theme_path = [wild_sphinx_theme.get_theme_dir()]

# html_title = project + ' version ' + release

html_show_sourcelink = False

# html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'searchbox.html']}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------