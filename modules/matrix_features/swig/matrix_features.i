%module matrix_features
%{
    #define SWIG_FILE_WITH_INIT
    #include "FeatureExtractorStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <matrix_features_external.i>

%include "FeatureExtractorStep.h"

%template(AxisAlignedParamsStep_f32i32) AxisAlignedParamsStep<float, int>;
%template(DimensionPairDifferenceParamsStep_f32i32) DimensionPairDifferenceParamsStep<float, int>;
%template(ClassPairDifferenceParamsStep_f32i32) ClassPairDifferenceParamsStep<float, int>;
%template(LinearFloat32MatrixFeature_f32i32) LinearMatrixFeature< MatrixBufferTemplate<float>, float, int >;
%template(LinearFloat32MatrixFeatureExtractorStep_f32i32) FeatureExtractorStep< LinearMatrixFeature<MatrixBufferTemplate<float>, float, int> >;
