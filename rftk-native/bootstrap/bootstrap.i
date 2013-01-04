%module bootstrap

%{
    #define SWIG_FILE_WITH_INIT
    #include "bootstrap.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* vec, int dim)}

%include "bootstrap.h"

