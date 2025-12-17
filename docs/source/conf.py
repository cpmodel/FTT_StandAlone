# Configuration file for the Sphinx documentation builder.

import os
import sys
import subprocess

# -- Path setup --------------------------------------------------------------

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
source_dir = os.path.join(project_root, 'SourceCode')

sys.path.insert(0, project_root)
sys.path.insert(0, source_dir)

# -- Project information -----------------------------------------------------

project = 'Future Technology Transformation (FTT)'
copyright = '2022, Rosie Hayward'
author = 'Rosie Hayward'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Automatic API doc generation --------------------------------------------

api_dir = os.path.join(os.path.dirname(__file__), 'api')
os.makedirs(api_dir, exist_ok=True)

# Generate .rst files for all subfolders in SourceCode
for entry in os.listdir(source_dir):
    full_path = os.path.join(source_dir, entry)
    if os.path.isdir(full_path) and not entry.startswith('__'):
        tocfile = f"{entry}_modules"
        subprocess.call([
            'sphinx-apidoc', '-f', '-o', api_dir, full_path, '--separate', '--tocfile', tocfile
        ])

# Generate modules.rst including all API .rst files
modules_rst_path = os.path.join(api_dir, 'modules.rst')
with open(modules_rst_path, 'w') as f:
    f.write("API Documentation\n")
    f.write("=================\n\n")
    f.write(".. toctree::\n")
    f.write("   :maxdepth: 2\n\n")
    for rst_file in sorted(os.listdir(api_dir)):
        if rst_file.endswith('.rst') and rst_file != 'modules.rst':
            name = os.path.splitext(rst_file)[0]
            f.write(f"   {name}\n")

