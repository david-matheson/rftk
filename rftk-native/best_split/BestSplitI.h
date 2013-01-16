#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;
class BufferCollection;

// todo: Old verision... need to fixup implementations
// class BestSplitI {
// public:
//   virtual void BestSplits(  BufferCollection& data,
//                             const MatrixBufferInt& sampleIndices,
//                             const MatrixBufferFloat& featureValues,
//                             MatrixBufferFloat& impurityOut,
//                             MatrixBufferFloat& thresholdOut) {}
// };

class BestSplitI //Already exists
{
public:
    virtual int GetYDim() const { return 1; }

    virtual void BestSplits( BufferCollection& data,
                            // const MatrixBufferInt& sampleIndices,
                            // const MatrixBufferFloat& featureValues, // contained in data (if needed)
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const {}
};

