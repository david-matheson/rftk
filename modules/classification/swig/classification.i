%module classification
%{
    #define SWIG_FILE_WITH_INIT
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
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"

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

%template(ClassInfoGainWalker_f32i32) ClassInfoGainWalker<float, int>;
%template(ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32) BestSplitpointsWalkingSortedStep< ClassInfoGainWalker<float, int> >;
%template(ClassInfoGainSplitpointsImpurity_f32i32) SplitpointsImpurity< ClassInfoGainImpurity<float>, int >;
%template(FinalizerI_f32) FinalizerI<float>;
%template(ClassEstimatorFinalizer_f32) ClassEstimatorFinalizer<float>;
%template(ClassProbabilityCombiner_f32) ClassProbabilityCombiner<float>;
%template(ClassStatsUpdater_f32i32) ClassStatsUpdater<float,int>;
%template(ClassStatsUpdaterOneStreamStep_f32i32) SplitpointStatsStep< ClassStatsUpdater<float,int> >;
%template(ClassStatsUpdaterTwoStreamStep_f32i32) TwoStreamSplitpointStatsStep< ClassStatsUpdater<float,int> >;