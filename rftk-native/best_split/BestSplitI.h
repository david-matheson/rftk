#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;
class BufferCollection;

class BestSplitI {
public:
  virtual void BestSplits(  BufferCollection& data,
                            const MatrixBufferInt& sampleIndices,
                            const MatrixBufferFloat& featureValues,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut) {}
};
