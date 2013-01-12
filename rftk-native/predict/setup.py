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
                       "VecPredict.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers", "-I../forest_data"],
                       include_dirs = ["../assert_util", "../buffers", "../features", "../forest_data"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_features.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_forest_data.so')],
                       )
elif sys.platform == 'darwin':
    _predict = Extension("_predict",
                       ["predict.i",
                       "VecPredict.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers", "-I../forest_data"],
                       include_dirs = ["../assert_util", "../buffers", "../features", "../forest_data"],
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

