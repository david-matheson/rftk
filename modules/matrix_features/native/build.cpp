#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "FeatureExtractorStep.h"
#include "AxisAlignedParamsStep.h"
#include "DimensionPairDifferenceParamsStep.h"
#include "ClassPairDifferenceParamsStep.h"
#include "LinearMatrixFeature.h"
#include "LinearMatrixFeatureBinding.h"

template class AxisAlignedParamsStep< DefaultBufferTypes >;
// template class LinearMatrixFeature< DefaultBufferTypes, MatrixBufferTemplate<float> >;
template class FeatureExtractorStep< LinearMatrixFeature<DefaultBufferTypes, MatrixBufferTemplate<DefaultBufferTypes::SourceContinuous > > >;

