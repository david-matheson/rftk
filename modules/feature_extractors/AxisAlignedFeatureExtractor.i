%{
  #include "AxisAlignedFeatureExtractor.h"
%}

%include "AxisAlignedFeatureExtractor.h"

%template(Float32AxisAlignedFeatureExtractorType)  AxisAlignedFeatureExtractor<Float32MatrixBuffer>;
%template(Float64AxisAlignedFeatureExtractorType)  AxisAlignedFeatureExtractor<Float64MatrixBuffer>;
%template(Float32SparseAxisAlignedFeatureExtractorType)  AxisAlignedFeatureExtractor<Float32SparseMatrixBuffer>;
%template(Float64SparseAxisAlignedFeatureExtractorType)  AxisAlignedFeatureExtractor<Float64SparseMatrixBuffer>;
