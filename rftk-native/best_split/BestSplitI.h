#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;
class BufferCollection;


class BestSplitI //Already exists
{
public:
    virtual int GetYDim() const { return 0; }

    virtual void BestSplits( BufferCollection& data,
                            // const MatrixBufferInt& sampleIndices,
                            // const MatrixBufferFloat& featureValues, // contained in data (if needed)
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const {}
};

