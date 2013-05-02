%module pipeline
%{
    #define SWIG_FILE_WITH_INIT
    #include "UniqueBufferId.h"
    #include "PipelineStepI.h"
    #include "Pipeline.h"
    #include "AllSamplesStep.h"
    #include "BootstrapSamplesStep.h"
    #include "SetBufferStep.h"
    #include "SliceBufferStep.h"
    #include "FeatureExtractorStep.h"
%}

%include <exception.i>
%import(module="rftk.asserts") "asserts.i"
%import(module="rftk.buffers") "buffers.i"
%include <pipeline_external.i>

%include "std_vector.i"

namespace std {
    %template(PipelineVector) std::vector<PipelineStepI*>;
}

%include "UniqueBufferId.h"
%include "PipelineStepI.h"
%include "Pipeline.h"
%include "AllSamplesStep.h"
%include "BootstrapSamplesStep.h"
%include "SetBufferStep.h"
%include "SliceBufferStep.h"
%include "FeatureExtractorStep.h"

%template(AllSamplesStep_f32f32i32) AllSamplesStep< MatrixBufferTemplate<float>, float, int >;
%template(AllSamplesStep_i32f32i32) AllSamplesStep< MatrixBufferTemplate<int>, float, int >;
%template(BootstrapSamplesStep_f32f32i32) BootstrapSamplesStep< MatrixBufferTemplate<float>, float, int >;
%template(BootstrapSamplesStep_i32f32i32) BootstrapSamplesStep< MatrixBufferTemplate<int>, float, int >;

%template(SetFloat32VectorBufferStep) SetBufferStep< VectorBufferTemplate<float> >;
%template(SetFloat64VectorBufferStep) SetBufferStep< VectorBufferTemplate<double> >;
%template(SetInt32VectorBufferStep) SetBufferStep< VectorBufferTemplate<int> >;
%template(SetInt64VectorBufferStep) SetBufferStep< VectorBufferTemplate<long long> >;

%template(SetFloat32MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<float> >;
%template(SetFloat64MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<double> >;
%template(SetInt32MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<int> >;
%template(SetInt64MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<long long> >;

%template(SliceFloat32MatrixBufferStep_i32) SliceBufferStep< MatrixBufferTemplate<float>, int >;
%template(SliceFloat32VectorBufferStep_i32) SliceBufferStep< VectorBufferTemplate<float>, int >;
%template(SliceInt32VectorBufferStep_i32) SliceBufferStep< VectorBufferTemplate<int>, int >;