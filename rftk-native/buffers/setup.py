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
      file_util.move_file("_buffers.so", os.path.expandvars('$PYTHONPATH/rftk/native'))
      file_util.move_file("buffers.py", os.path.expandvars('$PYTHONPATH/rftk/native'))


# extension module
# extension module
if sys.platform == 'linux2':
    _buffers = Extension("_buffers",
                       ["buffers.i",
                       "Tensor3Buffer.cpp",
                       "MatrixBuffer.cpp",
                       "BufferCollection.cpp"],
                       swig_opts=["-c++", "-I../assert_util"],
                       include_dirs = [numpy_include, "../assert_util"],
                       runtime_library_dirs = [os.path.expandvars('$PYTHONPATH/rftk/')],
                       extra_objects = [os.path.expandvars('$PYTHONPATH/rftk/native/_assert_util.so')],
                       )
elif sys.platform == 'darwin':
    _buffers = Extension("_buffers",
                       ["buffers.i",
                       "Tensor3Buffer.cpp",
                       "MatrixBuffer.cpp",
                       "BufferCollection.cpp"],
                       swig_opts=["-c++", "-I../assert_util"],
                       include_dirs = [numpy_include, "../assert_util"],
                       )

# NumyTypemapTests setup
setup(  name        = "buffers",
        description = "C++ helpers for wrapping numpy buffers.  This simplifies the rest of the code",
        author      = "David Matheson",
        version     = "1.0",
        ext_modules = [_buffers],
        py_modules = ["buffers"],
        cmdclass={'rftkinstall': my_install}
        )

