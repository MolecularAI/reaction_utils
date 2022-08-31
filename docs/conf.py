import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = "ReactionUtils"
copyright = "2022, Molecular AI group"
author = "Molecular AI group"
release = "1.1.1"

extensions = [
    "sphinx.ext.autodoc",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "alabaster"
html_theme_options = {
    "description": "Utilities for working with reactions, reaction templates and template extraction",
    "fixed_sidebar": True,
}
