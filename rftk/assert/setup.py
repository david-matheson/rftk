#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# extension module
_assert = Extension("_assert",
                   ["assert.i",
                   "assert.cpp"],
                   swig_opts=["-c++"],
                   )

# NumyTypemapTests setup
setup(  name        = "assert",
        description = "common assert macros",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_assert],
        py_modules = ["assert"],
        )

