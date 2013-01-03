#pragma once

#include "MatrixBuffer.h"

#include "FeatureExtractorI.h"
#include "FeatureTypes.h"


class AxisAlignedFeatureExtractor : public FeatureExtractorI {
public:
  AxisAlignedFeatureExtractor( const MatrixBufferFloat& xs );

  ~AxisAlignedFeatureExtractor() {}

  virtual int GetUID() { return STANDARD_FEATURE_AXIS_ALIGNED; }

  virtual void Extract( const MatrixBufferInt& sampleIndices,
                        const MatrixBufferInt& intFeatureParams,
                        const MatrixBufferFloat& floatFeatureParams,
                        MatrixBufferFloat& featureValuesOUT); // #tests X #samples

  MatrixBufferFloat mXs;
};
