%module classification
%{
    #define SWIG_FILE_WITH_INIT
    #include "ClassInfoGainWalker.h"
    #include "BestSplitpointsWalkingSortedStep.h"
    #include "FinalizerI.h"
    #include "ClassEstimatorFinalizer.h"
    #include "ClassProbabilityCombiner.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%import(module="rftk.splitpoints") "splitpoints.i"

%include "ClassInfoGainWalker.h"
%include "BestSplitpointsWalkingSortedStep.h"
%include "FinalizerI.h"
%include "ClassEstimatorFinalizer.h"
%include "ClassProbabilityCombiner.h"

%template(ClassInfoGainWalker_f32i32) ClassInfoGainWalker<float, int>;
%template(ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32) BestSplitpointsWalkingSortedStep< ClassInfoGainWalker<float, int> >;
%template(FinalizerI_f32) FinalizerI<float>;
%template(ClassEstimatorFinalizer_f32) ClassEstimatorFinalizer<float>;
%template(ClassProbabilityCombiner_f32) ClassProbabilityCombiner<float>;
