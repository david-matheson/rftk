%module feature_extractors
%{
    #define SWIG_FILE_WITH_INIT
    #include "FeatureExtractorI.h"
    #include "DepthScaledDepthDeltaFeatureExtractor.h"
    #include "DepthScaledEntangledYsFeatureExtractor.h"
%}

%include <exception.i>
%import "asserts/asserts.i"
%import "buffers/buffers.i"

%include "FeatureExtractorI.h"
%include "DepthScaledDepthDeltaFeatureExtractor.h"
%include "DepthScaledEntangledYsFeatureExtractor.h"

%include "AxisAlignedFeatureExtractor.i"
%include "RandomProjectionFeatureExtractor.i"
