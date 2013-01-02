#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# extension module
_features = Extension("_features",
                   ["features.i",
                   "AxisAlignedFeatureExtractor.cpp",
                   "ProjectionFeatureExtractor.cpp",
                   "DepthScaledDepthDeltaFeatureExtractor.cpp",
                   "DepthScaledEntangledYsFeatureExtractor.cpp",],
                   swig_opts=["-c++", "-I../assert_util", "-I../buffers"],
                    include_dirs = ["../assert_util", "../buffers"],
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

