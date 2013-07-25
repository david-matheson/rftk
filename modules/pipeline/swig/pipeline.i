%module pipeline
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "UniqueBufferId.h"
    #include "PipelineStepI.h"
    #include "Pipeline.h"
    #include "AllSamplesStep.h"
    #include "BootstrapSamplesStep.h"
    #include "PoissonSamplesStep.h"
    #include "PoissonStep.h"
    #include "SetBufferStep.h"
    #include "SliceBufferStep.h"
    #include "FeatureExtractorStep.h"
    #include "FeatureEqualI.h"
    #include "FeatureEqualQuantized.h"
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
%include "PoissonSamplesStep.h"
%include "PoissonStep.h"
%include "SetBufferStep.h"
%include "SliceBufferStep.h"
%include "FeatureExtractorStep.h"
%include "FeatureEqualI.h"
%include "FeatureEqualQuantized.h"

%template(AllSamplesStep_f32f32i32) AllSamplesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(AllSamplesStep_i32f32i32) AllSamplesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceInteger> >;
%template(BootstrapSamplesStep_f32f32i32) BootstrapSamplesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(BootstrapSamplesStep_i32f32i32) BootstrapSamplesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceInteger> >;
%template(PoissonSamplesStep_f32i32) PoissonSamplesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(PoissonSamplesStep_i32i32) PoissonSamplesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceInteger> >;

%template(PoissonStep_f32i32) PoissonStep< DefaultBufferTypes >;


%template(SetContinuousVectorBufferStep) SetBufferStep< VectorBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(SetIntegerVectorBufferStep) SetBufferStep< VectorBufferTemplate<DefaultBufferTypes::SourceInteger> >;
%template(SetContinuousMatrixBufferStep) SetBufferStep< MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(SetIntegerMatrixBufferStep) SetBufferStep< MatrixBufferTemplate<DefaultBufferTypes::SourceInteger> >;

%template(SetFloat32VectorBufferStep) SetBufferStep< VectorBufferTemplate<float> >;
%template(SetFloat64VectorBufferStep) SetBufferStep< VectorBufferTemplate<double> >;
%template(SetInt32VectorBufferStep) SetBufferStep< VectorBufferTemplate<int> >;
%template(SetInt64VectorBufferStep) SetBufferStep< VectorBufferTemplate<long long> >;

%template(SetFloat32MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<float> >;
%template(SetFloat64MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<double> >;
%template(SetInt32MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<int> >;
%template(SetInt64MatrixBufferStep) SetBufferStep< MatrixBufferTemplate<long long> >;

%template(SliceFloat32MatrixBufferStep_i32) SliceBufferStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(SliceFloat32VectorBufferStep_i32) SliceBufferStep< DefaultBufferTypes, VectorBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(SliceInt32VectorBufferStep_i32) SliceBufferStep< DefaultBufferTypes, VectorBufferTemplate<DefaultBufferTypes::SourceInteger> >;

%template(FeatureEqualI_f32i32) FeatureEqualI< DefaultBufferTypes >;
%template(FeatureEqualQuantized_f32i32) FeatureEqualQuantized< DefaultBufferTypes >;