%module pipeline
%{
    #define SWIG_FILE_WITH_INIT
    #include "FeatureOrdering.h"
    #include "UniqueBufferId.h"
    #include "PipelineStepI.h"
    #include "FeatureInfoLoggerI.h"
    #include "FeatureExtractorStep.h"
    #include "FeatureEqualI.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"

%include "FeatureOrdering.h"
%include "UniqueBufferId.h"
%include "PipelineStepI.h"
%include "FeatureInfoLoggerI.h"
%include "FeatureExtractorStep.h"
%include "FeatureEqualI.h"

