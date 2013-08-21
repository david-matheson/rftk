%module image_features
%{
    #define SWIG_FILE_WITH_INIT
    #include "PixelPairGaussianOffsetsStep.h"
    #include "ScaledDepthDeltaFeature.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%include "PixelPairGaussianOffsetsStep.h"
%include "ScaledDepthDeltaFeature.h"
