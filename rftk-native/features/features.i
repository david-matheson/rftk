%module features
%{
    #define SWIG_FILE_WITH_INIT
    #include "ImgFeatures.h"
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"

%include "ImgFeatures.h"

