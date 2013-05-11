%module splitpoints
%{
    #define SWIG_FILE_WITH_INIT
    #include "RandomSplitpointsStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <splitpoints_external.i>

%include "RandomSplitpointsStep.h"

%include "std_vector.i"

namespace std {
    %template(SplitSelectorBufferVector) std::vector<SplitSelectorBuffers>;
}

%template(SplitSelector_f32i32) SplitSelector<float, int>;
%template(RandomSplitpointsStep_f32i32) RandomSplitpointsStep<float, int>;