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
      file_util.move_file("_online.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("online.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
if sys.platform == 'linux2':
    _online = Extension("_online",
                       ["online.i",
                       "OnlineLeafFactories.cpp",
                       "AllThresholdsOnlineLeaf.cpp",
                       "RandomThresholdsOnlineLeaf.cpp",
                       "AlphaBetaSplitCriteria.cpp",
                       "HoeffdingSplitCriteria.cpp",
                       "OnlineLeafSet.cpp",
                       "OnlineTree.cpp"],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../bootstrap"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_bootstrap.so')],
                       )
elif sys.platform == 'darwin':
    _online = Extension("_online",
                       ["online.i",
                       "OnlineLeafFactories.cpp",
                       "AllThresholdsOnlineLeaf.cpp",
                       "RandomThresholdsOnlineLeaf.cpp",
                       "AlphaBetaSplitCriteria.cpp",
                       "HoeffdingSplitCriteria.cpp",
                       "OnlineLeafSet.cpp",
                       "OnlineTree.cpp"],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../bootstrap"],
                       )

# NumyTypemapTests setup
setup(  name        = "online",
        description = "Leaf node containers for online random forests",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_online],
        py_modules = ["online"],
        cmdclass={'rftkinstall': my_install}
        )

