# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information\

# Add package to path
from pathlib import Path
import sys

sys.path.insert(0, str(Path("../src/kmeans-tjdwill").resolve()))
import kmeans

project = 'K-Means Clustering'
copyright = '2024, Terrance Williams (tjdwill)'
author = 'Terrance Williams (tjdwill)'
release = kmeans.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

python_maximum_signature_line_length = 88


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

themes = {
    "material": "sphinx_material",
    "readthedocs": "sphinx_rtd_theme",
    "furo": "furo",
    "python": "python_docs_theme"
}
html_theme = themes['furo']
html_static_path = ['_static']

