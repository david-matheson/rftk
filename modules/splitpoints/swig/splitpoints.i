%module splitpoints
%{
    #define SWIG_FILE_WITH_INIT
    #include "RandomSplitpointsStep.h"
    #include "AssignStreamStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <splitpoints_external.i>

%include "RandomSplitpointsStep.h"
%include "AssignStreamStep.h"

%include "std_vector.i"

namespace std {
    %template(SplitSelectorBufferVector) std::vector<SplitSelectorBuffers>;
}

%template(SplitSelectorI_f32i32) SplitSelectorI<float, int>;
%template(SplitSelector_f32i32) SplitSelector<float, int>;
%template(WaitForBestSplitSelector_f32i32) WaitForBestSplitSelector<float, int>;
%template(RandomSplitpointsStep_f32i32) RandomSplitpointsStep<float, int>;
%template(AssignStreamStep_f32i32) AssignStreamStep<float, int>;