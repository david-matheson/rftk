%module learn
%{
    #define SWIG_FILE_WITH_INIT
    #include "TreeLearnerI.h"
    #include "DepthFirstTreeLearner.h"
    #include "ParallelForestLearner.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%include "TreeLearnerI.h"
%include "DepthFirstTreeLearner.h"
%include "ParallelForestLearner.h"

%template(DepthFirstTreeLearner_f32i32) DepthFirstTreeLearner<float, int>;
