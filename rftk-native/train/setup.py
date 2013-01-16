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
      file_util.move_file("_train.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("train.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
if sys.platform == 'linux2':
    _train = Extension("_train",
                       ["train.i",
                       "ActiveSplitNode.cpp",
                       "TrainConfigParams.cpp",
                       "DepthFirstParallelForestLearner.cpp",
                       "OnlineForestLearner.cpp",
                       "AllNodeDataCollector.cpp",
                       "OfflineSplitCriteria.cpp"],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers", 
                                    "-I../feature_extractors", "-I../forest_data"],
                       include_dirs = ["../assert_util", "../buffers", "../bootstrap",
                                      "../feature_extractors", "../best_split", "../forest_data", "../predict"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_bootstrap.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_feature_extractors.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_best_split.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_forest_data.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_predict.so')]

                       )
elif sys.platform == 'darwin':
    _train = Extension("_train",
                       ["train.i",
                       "ActiveSplitNode.cpp",
                       "TrainConfigParams.cpp",
                       "DepthFirstParallelForestLearner.cpp",
                       "OnlineForestLearner.cpp",
                       "AllNodeDataCollector.cpp",
                       "OfflineSplitCriteria.cpp"],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers", 
                                    "-I../feature_extractors", "-I../forest_data"],
                       include_dirs = ["../assert_util", "../buffers", "../bootstrap",
                                      "../feature_extractors", "../best_split", "../forest_data", "../predict"],
                       )

# NumyTypemapTests setup
setup(  name        = "train",
        description = "Code for training offline, online and entangled random forests",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_train],
        py_modules = ["train"],
        cmdclass={'rftkinstall': my_install}
        )

