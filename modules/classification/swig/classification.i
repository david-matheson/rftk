%module classification
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "SplitpointsImpurity.h"
    #include "SplitpointStatsStep.h"
    #include "TwoStreamSplitpointStatsStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"
%include <classification_external.i>

%include "SplitpointsImpurity.h"
%include "SplitpointStatsStep.h"
%include "TwoStreamSplitpointStatsStep.h"

%template(ClassInfoGainWalker_f32i32) ClassInfoGainWalker< DefaultBufferTypes >;
%template(ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32) BestSplitpointsWalkingSortedStep< ClassInfoGainWalker<DefaultBufferTypes> >;
%template(ClassInfoGainSplitpointsImpurity_f32i32) SplitpointsImpurity< ClassInfoGainImpurity<DefaultBufferTypes> >;

%template(ClassEstimatorFinalizer_f32) ClassEstimatorFinalizer<DefaultBufferTypes>;
%template(ClassEstimatorUpdater_f32i32) ClassEstimatorUpdater<float, int>;
%template(ClassProbabilityCombiner_f32) ClassProbabilityCombiner<float>;
%template(ClassStatsUpdater_f32i32) ClassStatsUpdater<float,int>;
%template(ClassStatsUpdaterOneStreamStep_f32i32) SplitpointStatsStep< ClassStatsUpdater<float,int> >;
%template(ClassStatsUpdaterTwoStreamStep_f32i32) TwoStreamSplitpointStatsStep< ClassStatsUpdater<float,int> >;