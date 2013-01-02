#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig
from distutils      import file_util
from distutils.command.install import install

import os
import sys

class my_install(install):
    def run(self):
      file_util.move_file("_assert_util.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("assert_util.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

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
        cmdclass={'rftkinstall': my_install}
        )

