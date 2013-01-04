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
      file_util.move_file("_feature_extractors.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("feature_extractors.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
if sys.platform == 'linux2':
    _feature_extractors = Extension("_feature_extractors",
                       ["feature_extractors.i",
                       "AxisAlignedFeatureExtractor.cpp",
                       "ProjectionFeatureExtractor.cpp",
                       "DepthScaledDepthDeltaFeatureExtractor.cpp",
                       "DepthScaledEntangledYsFeatureExtractor.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../features"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_buffers.so'),
                                        os.path.expandvars('$PYTHONPATH/rftk/native/_features.so')],
                       )
elif sys.platform == 'darwin':
    _feature_extractors = Extension("_feature_extractors",
                       ["feature_extractors.i",
                       "AxisAlignedFeatureExtractor.cpp",
                       "ProjectionFeatureExtractor.cpp",
                       "DepthScaledDepthDeltaFeatureExtractor.cpp",
                       "DepthScaledEntangledYsFeatureExtractor.cpp",],
                       swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                       include_dirs = ["../assert_util", "../buffers", "../features"],
                       )

# NumyTypemapTests setup
setup(  name        = "feature_extractors",
        description = "Extract 1d feature_extractors from the dataset.  \
                        In standard random forests these would be axis algined.",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_feature_extractors],
        py_modules = ["feature_extractors"],
        cmdclass={'rftkinstall': my_install}
        )

