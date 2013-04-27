%module matrix_features
%{
    #define SWIG_FILE_WITH_INIT
    #include "AxisAlignedParamsStep.h"
    #include "LinearMatrixFeature.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%include <pipeline.i>

%include "AxisAlignedParamsStep.h"
%include "LinearMatrixFeature.h"

%template(LinearMatrixFeature_f32i32) LinearMatrixFeature< MatrixBufferTemplate<float>, float, int >;