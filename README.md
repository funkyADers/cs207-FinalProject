# cs207-FinalProject
CS207 final project

[![Build Status](https://travis-ci.org/funkyADers/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/funkyADers/cs207-FinalProject.svg?branch=master)

[![Coverage Status](https://codecov.io/gh/funkyADers/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/funkyADers/cs207-FinalProject)

Installation instructions:

- Make a new conda virtual environment and change into it

- pip install -r requirements.txt

- pip install -i https://test.pypi.org/simple/ funkyAD-funkyADers==0.0.3

Releasing a new version of the package instructions:

- Change setup.py to hold new version number

- python3 setup.py sdist bdist_wheel

- python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

- user: tyleryoo pass: sasmwng5

- update the version number in README and docs

- make sure to uninstall the old version (pip uninstall funkyAD-funkyADers) and reinstall the new one before testing

Group no. 4
Anna Zink
Johannes K. Kolberg
Fabio Pruneri
Tyler Yoo
