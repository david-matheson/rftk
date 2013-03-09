%{
  #include "RandomProjectionFeatureExtractor.h"
%}

%include "RandomProjectionFeatureExtractor.h"

%template(Float32RandomProjectionFeatureExtractorType)  RandomProjectionFeatureExtractor<Float32MatrixBuffer>;
%template(Float64RandomProjectionFeatureExtractorType)  RandomProjectionFeatureExtractor<Float64MatrixBuffer>;
%template(Float32SparseRandomProjectionFeatureExtractorType)  RandomProjectionFeatureExtractor<Float32SparseMatrixBuffer>;
%template(Float64SparseRandomProjectionFeatureExtractorType)  RandomProjectionFeatureExtractor<Float64SparseMatrixBuffer>;
