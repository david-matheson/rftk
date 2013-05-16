%module image_features
%{
    #define SWIG_FILE_WITH_INIT
    #include "FeatureExtractorStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <image_features_external.i>

%include "FeatureExtractorStep.h"

%template(PixelPairGaussianOffsetsStep_f32i32) PixelPairGaussianOffsetsStep<float, int>;
%template(ScaledDepthDeltaFeature_f32i32) ScaledDepthDeltaFeature< float, int >;
%template(ScaledDepthDeltaFeatureExtractorStep_f32i32) FeatureExtractorStep< ScaledDepthDeltaFeature<float, int> >;