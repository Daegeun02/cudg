#!/bin/bash

rm -rf ./build
rm -rf ./dist
rm -rf ./Pygeun.egg-info

python3 setup.py sdist bdist_wheel

python3 -m twine upload ./dist/*