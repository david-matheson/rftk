#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;

class BestSplitI {
public:
  virtual void BestSplits(  const MatrixBufferInt& sampleIndices,
                            const MatrixBufferFloat& featureValues,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut) {}
};
