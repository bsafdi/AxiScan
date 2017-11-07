#!/bin/bash

# Script to compile cython code

cd Data_Generation/cython/
./make.sh
cd ..
cd ..

cd Coarse_Scan/cython/
./make.sh
cd ..
cd ..

cd MultiNestScan/cython/
./make.sh
cd ..
cd ..


