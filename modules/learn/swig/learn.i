%module learn
%{
    #define SWIG_FILE_WITH_INIT
    #include "TreeLearnerI.h"
    #include "DepthFirstTreeLearner.h"
    #include "ParallelForestLearner.h"


    #include "OnlineForestLearner.h"
    #include "AxisAlignedParamsStep.h"
    #include "LinearMatrixFeature.h"
    #include "ClassEstimatorUpdater.h"
    #include "ClassProbabilityOfError.h"
    #include "SplitSelectorBuffers.h"
    #include "SplitSelectorI.h"
    #include "SplitSelector.h"
    #include "WaitForBestSplitSelector.h"
    #include "SplitSelectorInfo.h"
    #include "SplitpointStatsStep.h"
    #include "SplitpointsImpurity.h"
    #include "SplitpointStatsStep.h"
    #include "TwoStreamSplitpointStatsStep.h"
    #include "ClassInfoGainWalker.h"
    #include "ClassInfoGainImpurity.h"
    #include "BestSplitpointsWalkingSortedStep.h"
    #include "FinalizerI.h"
    #include "ClassEstimatorFinalizer.h"
    #include "ClassProbabilityCombiner.h"
    #include "ClassStatsUpdater.h"
    #include "SplitpointsImpurity.h"
    #include "SplitpointStatsStep.h"
    #include "TwoStreamSplitpointStatsStep.h"
    #include "RandomSplitpointsStep.h"
    #include "AssignStreamStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%import(module="rftk.matrix_features") "matrix_features.i"
%import(module="rftk.classification") "classification.i"
%import(module="rftk.splitpoints") "splitpoints.i"

%include "TreeLearnerI.h"
%include "DepthFirstTreeLearner.h"
%include "ParallelForestLearner.h"
%include "OnlineForestLearner.h"

%template(DepthFirstTreeLearner_f32i32) DepthFirstTreeLearner<float, int>;

%include "AxisAlignedParamsStep.h"
%include "LinearMatrixFeature.h"
%include "ClassEstimatorUpdater.h"
%include "ClassProbabilityOfError.h"
%include "SplitSelectorBuffers.h"
%include "SplitSelectorI.h"
%include "SplitSelector.h"
%include "WaitForBestSplitSelector.h"
%include "SplitSelectorInfo.h"
%include "SplitpointStatsStep.h"
%include "ClassInfoGainWalker.h"
%include "ClassInfoGainImpurity.h"
%include "BestSplitpointsWalkingSortedStep.h"
%include "FinalizerI.h"
%include "ClassEstimatorFinalizer.h"
%include "ClassProbabilityCombiner.h"
%include "ClassStatsUpdater.h"
%include "SplitpointsImpurity.h"
%include "SplitpointStatsStep.h"
%include "TwoStreamSplitpointStatsStep.h"
%include "RandomSplitpointsStep.h"
%include "AssignStreamStep.h"

%template(OnlineForestMatrixClassLearner_f32i32)  OnlineForestLearner< LinearMatrixFeature< MatrixBufferTemplate<float>, float, int >, ClassEstimatorUpdater< float, int >, ClassProbabilityOfError, float, int >;
