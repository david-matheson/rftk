%module predict
%{
    #define SWIG_FILE_WITH_INIT
    #include "ForestPredictor.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.forest_data") "forest_data.i"

%include "ForestPredictor.h"
