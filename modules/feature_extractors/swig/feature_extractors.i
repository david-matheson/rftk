%module feature_extractors
%{
    #define SWIG_FILE_WITH_INIT
    #include "FeatureExtractorI.h"
    #include "DepthScaledDepthDeltaFeatureExtractor.h"
    #include "DepthScaledEntangledYsFeatureExtractor.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"

%include "FeatureExtractorI.h"
%include "DepthScaledDepthDeltaFeatureExtractor.h"

%include "AxisAlignedFeatureExtractor.i"
%include "RandomProjectionFeatureExtractor.i"
