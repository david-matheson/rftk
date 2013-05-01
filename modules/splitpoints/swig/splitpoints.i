%module splitpoints
%{
    #define SWIG_FILE_WITH_INIT
    #include "SplitSelectorBuffers.h"
    #include "SplitSelector.h"
    #include "SplitSelectorInfo.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"

%include "SplitSelectorBuffers.h"
%include "SplitSelector.h"
%include "SplitSelectorInfo.h"

%include "std_vector.i"

namespace std {
    %template(SplitSelectorBufferVector) std::vector<SplitSelectorBuffers>;
}

%template(SplitSelector_f32i32) SplitSelector<float, int>;