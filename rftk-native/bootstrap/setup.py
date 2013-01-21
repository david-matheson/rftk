#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig
from distutils      import file_util
from distutils.command.install import install

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

class my_install(install):
    def run(self):
      file_util.move_file("_bootstrap.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("bootstrap.py", os.path.expandvars('$PYTHONPATH/rftk/native'))

# extension module
_bootstrap = Extension("_bootstrap",
                   ["bootstrap.i","bootstrap.cpp"],
                   swig_opts=["-c++", "-I../assert_util"],
                   include_dirs = [numpy_include, "../assert_util"],
                   runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                   extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so')],
                   )

if sys.platform == 'linux2':
    _bootstrap = Extension("_bootstrap",
                       ["bootstrap.i","bootstrap.cpp"],
                       swig_opts=["-c++", "-I../assert_util"],
                       include_dirs = [numpy_include, "../assert_util"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so')],
                       )
elif sys.platform == 'darwin':
    _bootstrap = Extension("_bootstrap",
                       ["bootstrap.i","bootstrap.cpp"],
                       swig_opts=["-c++", "-I../assert_util"],
                       include_dirs = [numpy_include, "../assert_util"],
                       )


# NumyTypemapTests setup
setup(  name        = "function",
        description = "Generate bootstrap samples",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_bootstrap],
        cmdclass={'rftkinstall': my_install}
        )

