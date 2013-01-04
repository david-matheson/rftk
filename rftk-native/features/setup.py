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
      file_util.move_file("_features.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("features.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
if sys.platform == 'linux2':
    _features = Extension("_features",
                       ["features.i",
                       "ImgFeatures.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so')],
                       )
elif sys.platform == 'darwin':
    _features = Extension("_features",
                       ["features.i",
                       "ImgFeatures.cpp"],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers"],
                       )

# NumyTypemapTests setup
setup(  name        = "features",
        description = "Vec and img features",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_features],
        py_modules = ["features"],
        cmdclass={'rftkinstall': my_install}
        )

