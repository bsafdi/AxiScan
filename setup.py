# File required to compile the cython

import logging
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import cython_gsl

include_gsl_dir = "/sw/arcts/centos7/gsl/2.1/include/"
lib_gsl_dir = "/sw/arcts/centos7/gsl/2.1/lib/"


extensions = [
    Extension("ABRA-DATA.*", ["ABRA-DATA/*.pyx"],
        include_dirs=[numpy.get_include(),cython_gsl.get_cython_include_dir()], extra_compile_args=["-ffast-math",'-O3',"-march=native"], libraries=cython_gsl.get_libraries(),  library_dirs=[cython_gsl.get_library_dir()])
]


setup(
    ext_modules = cythonize(extensions),
)
