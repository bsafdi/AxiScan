#!/bin/bash

# Script to compile cython code

cd Data_Generation/
./make.sh
cd ..

cd Coarse_Scan/
./make.sh
cd ..

cd MultiNestScan/
./make.sh
cd ..


