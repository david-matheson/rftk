%module classification
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "SplitpointsImpurity.h"
    #include "SplitpointStatsStep.h"
    #include "TwoStreamSplitpointStatsStep.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
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
%template(ClassEstimatorUpdater_f32i32) ClassEstimatorUpdater<DefaultBufferTypes>;
%template(ClassProbabilityCombiner_f32) ClassProbabilityCombiner<DefaultBufferTypes>;
%template(ClassStatsUpdater_f32i32) ClassStatsUpdater<DefaultBufferTypes>;
%template(ClassStatsUpdaterOneStreamStep_f32i32) SplitpointStatsStep< ClassStatsUpdater<DefaultBufferTypes> >;
%template(ClassStatsUpdaterTwoStreamStep_f32i32) TwoStreamSplitpointStatsStep< ClassStatsUpdater<DefaultBufferTypes> >;