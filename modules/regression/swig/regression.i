%module regression
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
%include <regression_external.i>

%include "SplitpointsImpurity.h"
%include "SplitpointStatsStep.h"
%include "TwoStreamSplitpointStatsStep.h"



%template(SumOfVarianceWalker_f32i32) SumOfVarianceWalker<DefaultBufferTypes>;
%template(SumOfVarianceBestSplitpointsWalkingSortedStep_f32i32) BestSplitpointsWalkingSortedStep< SumOfVarianceWalker<DefaultBufferTypes> >;
%template(SumOfVarianceDownSampleBestSplitpointsWalkingSortedStep_f32i32) DownSampleBestSplitpointsWalkingSortedStep< SumOfVarianceWalker<DefaultBufferTypes> >;
%template(SumOfVarianceTwoStreamWalker_f32i32) SumOfVarianceTwoStreamWalker<DefaultBufferTypes>;
%template(SumOfVarianceTwoStreamBestSplitpointsWalkingSortedStep_f32i32) TwoStreamBestSplitpointsWalkingSortedStep< SumOfVarianceTwoStreamWalker<DefaultBufferTypes> >;
%template(SumOfVarianceRandomGapSplitpointsStep_f32i32) RandomGapSplitpointsStep< SumOfVarianceWalker<DefaultBufferTypes> >;

%template(MeanVarianceEstimatorFinalizer_f32) MeanVarianceEstimatorFinalizer< DefaultBufferTypes >;
%template(MeanVarianceEstimatorUpdater_f32) MeanVarianceEstimatorUpdater< DefaultBufferTypes >;
%template(MeanVarianceCombiner_f32) MeanVarianceCombiner<DefaultBufferTypes>;

%template(MeanVarianceStatsUpdater_f32i32) MeanVarianceStatsUpdater<DefaultBufferTypes>;
%template(SumOfVarianceOneStreamStep_f32i32) SplitpointStatsStep< MeanVarianceStatsUpdater<DefaultBufferTypes> >;
%template(SumOfVarianceTwoStreamStep_f32i32) TwoStreamSplitpointStatsStep< MeanVarianceStatsUpdater<DefaultBufferTypes> >;
%template(SumOfVarianceSplitpointsImpurity_f32i32) SplitpointsImpurity< SumOfVarianceImpurity<DefaultBufferTypes> >;