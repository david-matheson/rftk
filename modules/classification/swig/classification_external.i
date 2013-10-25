%module classification
%{
    #define SWIG_FILE_WITH_INIT
    #include "ClassInfoGainWalker.h"
    #include "ClassInfoGainTwoStreamWalker.h"
    #include "ClassInfoGainImpurity.h"
    #include "BestSplitpointsWalkingSortedStep.h"
    #include "DownSampleBestSplitpointsWalkingSortedStep.h"
    #include "TwoStreamBestSplitpointsWalkingSortedStep.h"
    #include "RandomGapSplitpointsStep.h"
    #include "FinalizerI.h"
    #include "ClassEstimatorFinalizer.h"
    #include "ClassEstimatorUpdater.h"
    #include "ClassProbabilityCombiner.h"
    #include "ClassStatsUpdater.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"

%include "ClassInfoGainWalker.h"
%include "ClassInfoGainTwoStreamWalker.h"
%include "ClassInfoGainImpurity.h"
%include "BestSplitpointsWalkingSortedStep.h"
%include "DownSampleBestSplitpointsWalkingSortedStep.h"
%include "TwoStreamBestSplitpointsWalkingSortedStep.h"
%include "RandomGapSplitpointsStep.h"
%include "FinalizerI.h"
%include "ClassEstimatorFinalizer.h"
%include "ClassEstimatorUpdater.h"
%include "ClassProbabilityCombiner.h"
%include "ClassStatsUpdater.h"