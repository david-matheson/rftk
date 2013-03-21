%module features
%{
    #define SWIG_FILE_WITH_INIT
    #include "ImgFeatures.h"
    #include "VecFeatures.h"
%}

%include <exception.i>
%import "asserts/asserts.i"
%import "buffers/buffers.i"

%include "ImgFeatures.h"
%include "VecFeatures.h"

