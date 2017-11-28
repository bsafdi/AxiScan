<<<<<<< HEAD
# Cython compilation
=======
# File required to compile the cython
>>>>>>> 6aa05aa94c3573cf3422b20a76120fb3e818b3c7

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
<<<<<<< HEAD
=======
#import cython_gsl
#import scipy.special as sc
>>>>>>> 6aa05aa94c3573cf3422b20a76120fb3e818b3c7

include_gsl_dir = "/sw/arcts/centos7/gsl/2.1/include/"
lib_gsl_dir = "/sw/arcts/centos7/gsl/2.1/lib/"

<<<<<<< HEAD
=======

# ext = Extension("sl", sources = ["gsl_test.pyx"],include_dirs=[numpy.get_include(),include_gsl_dir],library_dirs=[lib_gsl_dir],libraries=["gsl"])
#libraries=["gsl", "gslcblas"]
>>>>>>> 6aa05aa94c3573cf3422b20a76120fb3e818b3c7
extensions = [
    Extension("*", ["*.pyx"],
              include_dirs=[numpy.get_include(),include_gsl_dir], library_dirs=[lib_gsl_dir],libraries=["m","gsl","gslcblas"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp","-lcblas","-lgslcblas","-lgsl"],
              extra_link_args=['-fopenmp'])
<<<<<<< HEAD
]
setup(
    ext_modules = cythonize(extensions),
)
=======
        #extra_compile_args=["-ffast-math",'-O3']) 
]
setup(
    #name = "My hello app",
    ext_modules = cythonize(extensions),
)


# extensions = [
#     Extension("*", ["*.pyx"],
#               include_dirs=[numpy.get_include(),cython_gsl.get_cython_include_dir()], libraries=["m",cython_gsl.get_libraries()],library_dirs=[cython_gsl.get_library_dir()],
#               extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
#               extra_link_args=['-fopenmp'])
#         #extra_compile_args=["-ffast-math",'-O3']) 
# ]
# setup(
#     #name = "My hello app",
#     ext_modules = cythonize(extensions),
# )
>>>>>>> 6aa05aa94c3573cf3422b20a76120fb3e818b3c7
