#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# extension module
_buffers = Extension("_buffers",
                   ["buffers.i",
                   "ImgBuffer.cpp",
                   "MatrixBuffer.cpp"], #Link in assert logic
                   swig_opts=["-c++", "-I../assert_util"],
                   include_dirs = [numpy_include, "../assert_util"],
                   )

# NumyTypemapTests setup
setup(  name        = "buffers",
        description = "C++ helpers for wrapping numpy buffers.  This simplifies the rest of the code",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_buffers],
        py_modules = ["buffers"],
        )

