# celib

CE Python library

## Installation

From the command line:

	pip install .

If the version number has not changed, you may need to install with the
`--upgrade` option:

	pip install . --upgrade

## Dependencies

* `requests`
* `numpy`
* `pandas`
* `matplotlib`

## Test suite

From the current directory.

Using the batteries-included `unittest` module:

	python -m unittest discover .

Alternatively, with `nose` and `coverage` installed:

	nosetests -v --with-coverage --cover-package=celib
	coverage report -m
