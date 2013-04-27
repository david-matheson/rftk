%module pipeline
%{
    #define SWIG_FILE_WITH_INIT
    #include "PipelineStepI.h"
    #include "Pipeline.h"
    #include "SetBufferStep.h"
    #include "UniqueBufferId.h"
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
%include "SetBufferStep.h"
%include "UniqueBufferId.h"

%template(SetFloat32VectorBufferStep) SetBufferStep< VectorBufferTemplate<float> >;
%template(SetFloat64VectorBufferStep) SetBufferStep< VectorBufferTemplate<double> >;
%template(SetInt32VectorBufferStep) SetBufferStep< VectorBufferTemplate<int> >;
%template(SetInt64VectorBufferStep) SetBufferStep< VectorBufferTemplate<long long> >;

%template(SetFloat32MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<float> >;
%template(SetFloat64MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<double> >;
%template(SetInt32MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<int> >;
%template(SetInt64MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<long long> >;