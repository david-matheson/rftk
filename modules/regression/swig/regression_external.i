%module regression
%{
    #define SWIG_FILE_WITH_INIT
    #include "SumOfVarianceWalker.h"
    #include "SumOfVarianceTwoStreamWalker.h"
    #include "SumOfVarianceImpurity.h"
    #include "BestSplitpointsWalkingSortedStep.h"
    #include "DownSampleBestSplitpointsWalkingSortedStep.h"
    #include "TwoStreamBestSplitpointsWalkingSortedStep.h"
    #include "RandomGapSplitpointsStep.h"
    #include "FinalizerI.h"
    #include "MeanVarianceEstimatorFinalizer.h"
    #include "MeanVarianceEstimatorUpdater.h"
    #include "MeanVarianceCombiner.h"
    #include "MeanVarianceStatsUpdater.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"

%include "SumOfVarianceWalker.h"
%include "SumOfVarianceTwoStreamWalker.h"
%include "SumOfVarianceImpurity.h"
%include "BestSplitpointsWalkingSortedStep.h"
%include "DownSampleBestSplitpointsWalkingSortedStep.h"
%include "TwoStreamBestSplitpointsWalkingSortedStep.h"
%include "RandomGapSplitpointsStep.h"
%include "FinalizerI.h"
%include "MeanVarianceEstimatorFinalizer.h"
%include "MeanVarianceEstimatorUpdater.h"
%include "MeanVarianceCombiner.h"
%include "MeanVarianceStatsUpdater.h"