#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "FeatureExtractorStep.h"
#include "AxisAlignedParamsStep.h"
#include "DimensionPairDifferenceParamsStep.h"
#include "ClassPairDifferenceParamsStep.h"
#include "LinearMatrixFeature.h"
#include "LinearMatrixFeatureBinding.h"

template class AxisAlignedParamsStep<float, int>;
template class LinearMatrixFeature< MatrixBufferTemplate<float>, float, int >;
template class FeatureExtractorStep< LinearMatrixFeature<MatrixBufferTemplate<float>, float, int> >;