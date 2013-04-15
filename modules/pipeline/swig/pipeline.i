%module pipeline
%{
    #define SWIG_FILE_WITH_INIT
    #include "PipelineStepI.h"
    #include "Pipeline.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"

%include "std_vector.i"

namespace std {
    %template(PipelineVector) std::vector<PipelineStepI*>;
}

%include "PipelineStepI.h"
%include "Pipeline.h"