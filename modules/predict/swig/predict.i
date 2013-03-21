%module predict
%{
    #define SWIG_FILE_WITH_INIT
    #include "ForestPredictor.h"
%}

%include <exception.i>
%import "asserts.i"
%import "buffers.i"
%import "forest_data.i"

%include "ForestPredictor.h"
