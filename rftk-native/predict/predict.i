%module predict
%{
    #define SWIG_FILE_WITH_INIT
    #include
%}

%include <exception.i>
%import "assert_util.i"
%import "buffers.i"

%include "FeatureExtractorI.h"
%include "AxisAlignedFeatureExtractor.h"
%include "ProjectionFeatureExtractor.h"
%include "DepthScaledDepthDeltaFeatureExtractor.h"
%include "DepthScaledEntangledYsFeatureExtractor.h"
