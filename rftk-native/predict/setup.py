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
      file_util.move_file("_predict.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("predict.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
if sys.platform == 'linux2':
    _predict = Extension("_predict",
                       ["predict.i",
                       "Forest.cpp",
                       "VecPredict.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../features"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_features.so')],
                       )
elif sys.platform == 'darwin':
    _predict = Extension("_predict",
                       ["predict.i",
                       "Forest.cpp",
                       "VecPredict.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../features"],
                       )

# NumyTypemapTests setup
setup(  name        = "predict",
        description = "Oracle of the forest",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_predict],
        py_modules = ["predict"],
        cmdclass={'rftkinstall': my_install}
        )

