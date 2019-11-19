# cs207-FinalProject

[![Build Status](https://travis-ci.org/funkyADers/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/funkyADers/cs207-FinalProject.svg?branch=master)

[![Coverage Status](https://codecov.io/gh/funkyADers/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/funkyADers/cs207-FinalProject)

## funkyAD - an autodifferentiation package
**Group 4**: Anna Zink, Fabio Pruneri, Johannes K. Kolberg, Tyler Yoo

We recommend installing *funkyAD* in a virtual environment, as it depends on a specific version of *Numpy*.

To create a new virtual environment:
	
	conda create -n env_name python=3.7 anaconda
	source activate env_name

*funkyAD* can be installed either using `pip` with PyPi:
	
	pip install -i https://test.pypi.org/simple/ funkyAD-funkyADers

Or by cloning the source code and installing that directly:

	git clone git@github.com:funkyADers/cs207-FinalProject.git
	cd funkyAD
	pip install -r requirements.txt
	pip install -e .

The `-e` flag makes the source code editable without having to reinstall the package from the updated local source code every time. Handy if you wish to add custom functionality or build upon the *funkyAD* source code. Otherwise you can drop the `-e` flag, but note that you will still need the `.` (dot) to install the local source code via `setup.py` in the root directory.

See the docs for examples on using *funkyAD* in various applications.
