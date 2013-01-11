#pragma once

#include "MatrixBuffer.h"
#include "BufferCollection.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"


class AxisAlignedFeatureExtractor : public FeatureExtractorI {
public:
  AxisAlignedFeatureExtractor();

  ~AxisAlignedFeatureExtractor() {}

  virtual int GetUID() { return VEC_FEATURE_AXIS_ALIGNED; }

  virtual void Extract( BufferCollection& bufferCollection,
                        const MatrixBufferInt& sampleIndices,
                        const MatrixBufferInt& intFeatureParams,
                        const MatrixBufferFloat& floatFeatureParams,
                        MatrixBufferFloat& featureValuesOUT); // #tests X #samples
};
