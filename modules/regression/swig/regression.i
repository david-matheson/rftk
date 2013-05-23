%module regression
%{
    #define SWIG_FILE_WITH_INIT
    #include "SplitpointsImpurity.h"
    #include "SplitpointStatsStep.h"
    #include "TwoStreamSplitpointStatsStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%import(module="rftk.splitpoints") "splitpoints_external.i"
%include <regression_external.i>

%include "SplitpointsImpurity.h"
%include "SplitpointStatsStep.h"
%include "TwoStreamSplitpointStatsStep.h"

%template(MeanVarianceCombiner_f32) MeanVarianceCombiner<float>;
%template(SumOfVarianceWalker_f32i32) SumOfVarianceWalker<float, int>;
%template(SumOfVarianceBestSplitpointsWalkingSortedStep_f32i32) BestSplitpointsWalkingSortedStep< SumOfVarianceWalker<float, int> >;
%template(FinalizerI_reg_f32) FinalizerI<float>;
%template(MeanVarianceEstimatorFinalizer_f32) MeanVarianceEstimatorFinalizer< float >;

%template(SumOfVarianceTwoStreamBestSplitpointsWalkingSortedStep_f32i32) TwoStreamBestSplitpointsWalkingSortedStep< SumOfVarianceTwoStreamWalker<float, int> >;