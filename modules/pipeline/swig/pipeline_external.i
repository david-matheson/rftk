%module pipeline
%{
    #define SWIG_FILE_WITH_INIT
    #include "UniqueBufferId.h"
    #include "PipelineStepI.h"
    #include "FeatureIndexerI.h"
    #include "FeatureExtractorStep.h"
    #include "FeatureEqualI.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"

%include "UniqueBufferId.h"
%include "PipelineStepI.h"
%include "FeatureIndexerI.h"
%include "FeatureExtractorStep.h"
%include "FeatureEqualI.h"

