%module predict
%{
    #define SWIG_FILE_WITH_INIT
    #include "VecPredict.h"
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"



%include "VecPredict.h"
