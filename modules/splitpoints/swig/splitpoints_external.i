%module splitpoints
%{
    #define SWIG_FILE_WITH_INIT
    #include "FinalizerI.h"
    
    #include "SplitSelectorBuffers.h"

    #include "SplitSelectorI.h"
    #include "SplitSelector.h"
    #include "WaitForBestSplitSelector.h"
    #include "SplitSelectorInfo.h"
    #include "SplitpointStatsStep.h"

    #include "SplitpointsImpurity.h"
    #include "SplitpointStatsStep.h"
    #include "TwoStreamSplitpointStatsStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%include "FinalizerI.h"

%include "SplitSelectorBuffers.h"

%include "SplitSelectorI.h"
%include "SplitSelector.h"
%include "WaitForBestSplitSelector.h"
%include "SplitSelectorInfo.h"
%include "SplitpointStatsStep.h"

%include "SplitpointsImpurity.h"
%include "SplitpointStatsStep.h"
%include "TwoStreamSplitpointStatsStep.h"

%template(FinalizerI_f32) FinalizerI<DefaultBufferTypes>;