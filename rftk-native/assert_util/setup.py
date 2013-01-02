#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# sysconfig._config_vars['LDSHARED'] = 'g++ -shared'

# extension module
_assert_util = Extension("_assert_util",
                   ["assert_util.i",
                   "assert_util.cpp"],
                   swig_opts=["-c++"],
                   )

# NumyTypemapTests setup
setup(  name        = "assert_util",
        description = "common assert_util macros",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_assert_util],
        py_modules = ["assert_util"],
        )

