%module matrix_features
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "FeatureExtractorStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <matrix_features_external.i>

%include "FeatureExtractorStep.h"

%template(AxisAlignedParamsStep_f32i32) AxisAlignedParamsStep< DefaultBufferTypes >;
%template(DimensionPairDifferenceParamsStep_f32i32) DimensionPairDifferenceParamsStep< DefaultBufferTypes >;
%template(ClassPairDifferenceParamsStep_f32i32) ClassPairDifferenceParamsStep< DefaultBufferTypes >;

%template(LinearFloat32MatrixFeature_f32i32) LinearMatrixFeature< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(LinearFloat32MatrixFeatureExtractorStep_f32i32) FeatureExtractorStep< LinearMatrixFeature<DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> > >;
