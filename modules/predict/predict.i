%module predict
%{
    #define SWIG_FILE_WITH_INIT
    #include "ForestPredictor.h"
%}

%include <exception.i>
%import "asserts/asserts.i"
%import "buffers/buffers.i"
%import "forest_data/forest_data.i"

%include "ForestPredictor.h"
