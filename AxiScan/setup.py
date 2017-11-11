# Cython compilation

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

include_gsl_dir = "/sw/arcts/centos7/gsl/2.1/include/"
lib_gsl_dir = "/sw/arcts/centos7/gsl/2.1/lib/"

extensions = [
    Extension("*", ["*.pyx"],
              include_dirs=[numpy.get_include(),include_gsl_dir], library_dirs=[lib_gsl_dir],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp'])
]
setup(
    ext_modules = cythonize(extensions),
)
