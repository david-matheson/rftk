%module matrix_features
%{
    #define SWIG_FILE_WITH_INIT
    #include "BufferTypes.h"
    #include "FeatureExtractorStep.h"
    #include "SliceAxisAlignedFeaturesStep.h"
%}

%include <exception.i>
%import(module="rftk.utils") "utils.i"
%import(module="rftk.buffers") "buffers.i"
%import(module="rftk.pipeline") "pipeline_external.i"
%include <matrix_features_external.i>

%include "FeatureExtractorStep.h"
%include "SliceAxisAlignedFeaturesStep.h"

%template(AxisAlignedParamsStep_f32i32) AxisAlignedParamsStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(RandomProjectionParamsStep_f32i32) RandomProjectionParamsStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(DimensionPairDifferenceParamsStep_f32i32) DimensionPairDifferenceParamsStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(ClassPairDifferenceParamsStep_f32i32) ClassPairDifferenceParamsStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;

%template(AxisAlignedParamsStep_Sparse_f32i32) AxisAlignedParamsStep< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(RandomProjectionParamsStep_Sparse_f32i32) RandomProjectionParamsStep< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(DimensionPairDifferenceParamsStep_Sparse_f32i32) DimensionPairDifferenceParamsStep< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(ClassPairDifferenceParamsStep_Sparse_f32i32) ClassPairDifferenceParamsStep< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;

%template(LinearFloat32MatrixFeature_f32i32) LinearMatrixFeature< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(LinearFloat32MatrixFeatureExtractorStep_f32i32) FeatureExtractorStep< LinearMatrixFeature<DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> > >;
%template(SliceFloatMatrixFromAxisAlignedFeaturesStep_Default) SliceAxisAlignedFeaturesStep< DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;

%template(LinearFloat32MatrixFeature_Sparse_f32i32) LinearMatrixFeature< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;
%template(LinearFloat32MatrixFeatureExtractorStep_Sparse_f32i32) FeatureExtractorStep< LinearMatrixFeature<DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> > >;
%template(SliceFloatMatrixFromAxisAlignedFeaturesStep_Sparse_Default) SliceAxisAlignedFeaturesStep< DefaultBufferTypes, SparseMatrixBufferTemplate<DefaultBufferTypes::SourceContinuous> >;