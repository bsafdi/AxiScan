#!/bin/bash

# Script to compile cython code

rm *.so *.c
python setup.py build_ext --inplace
