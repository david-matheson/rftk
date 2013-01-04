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
      file_util.move_file("_best_split.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("best_split.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
if sys.platform == 'linux2':
    _best_split = Extension("_best_split",
                       ["best_split.i",
                       "Sorter.cpp",
                       "ClassInfoGainAllThresholdsBestSplit.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../bootstrap"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_bootstrap.so')],
                       )
elif sys.platform == 'darwin':
    _best_split = Extension("_best_split",
                       ["best_split.i",
                       "Sorter.cpp",
                       "ClassInfoGainAllThresholdsBestSplit.cpp"],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../bootstrap"],
                       )

# NumyTypemapTests setup
setup(  name        = "best_split",
        description = "Classes that find the feature and threshold that maximizes some impurity function",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_best_split],
        py_modules = ["best_split"],
        cmdclass={'rftkinstall': my_install}
        )

