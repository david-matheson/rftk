#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

import os

# extension module
_features = Extension("_features",
                   ["features.i",
                   "AxisAlignedFeatureExtractor.cpp",
                   "ProjectionFeatureExtractor.cpp",
                   "DepthScaledDepthDeltaFeatureExtractor.cpp",
                   "DepthScaledEntangledYsFeatureExtractor.cpp",],
                   swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                   include_dirs = ["../assert_util", "../buffers"],
                   runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                   extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/_assert_util.so'),
                                    os.path.expandvars('$PYTHONPATH/rftk/_buffers.so')],
                   )

# NumyTypemapTests setup
setup(  name        = "features",
        description = "Extract 1d features from the dataset.  \
                        In standard random forests these would be axis algined.",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_features],
        py_modules = ["features"],
        )

