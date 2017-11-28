from __future__ import print_function

import logging
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import cython_gsl

extensions = [
    Extension("AxiScan.axion_ll", ["AxiScan/axion_ll.pyx"],
              include_dirs=[numpy.get_include(),cython_gsl.get_include()], library_dirs=[cython_gsl.get_library_dir()],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp']),
    Extension("AxiScan.mc_gen", ["AxiScan/mc_gen.pyx"],
              include_dirs=[numpy.get_include(),cython_gsl.get_include()], library_dirs=[cython_gsl.get_library_dir()],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp']),
    Extension("AxiScan.scan", ["AxiScan/scan.pyx"],
              include_dirs=[numpy.get_include(),cython_gsl.get_include()], library_dirs=[cython_gsl.get_library_dir()],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp']),
    Extension("AxiScan.speed_dist", ["AxiScan/speed_dist.pyx"],
              include_dirs=[numpy.get_include(),cython_gsl.get_include()], library_dirs=[cython_gsl.get_library_dir()],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp'])
]

setup_args = {'name':'AxiScan',
    'version':'0.0',
    'description':'A Likelihood Framework for Axion Direct Detection',
    'url':'https://github.com/bsafdi/AxiScan',
    'author':'Benjamin R. Safdi',
    'author_email':'bsafdi@umich.edu',
    'license':'MIT',
    'install_requires':[
            'numpy',
            'matplotlib',
            'Cython',
            'pymultinest',
            'jupyter',
            'scipy',
            'CythonGSL',
        ]}

setup(packages=['AxiScan'],
    ext_modules = cythonize(extensions),
    **setup_args
)

print("Compilation successful!")
