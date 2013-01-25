#pragma once

class MatrixBufferInt;
class MatrixBufferFloat;
class BufferCollection;


class BestSplitI //Already exists
{
public:
    virtual BestSplitI* Clone() const { return NULL; }

    virtual int GetYDim() const { return 0; }

    virtual void BestSplits( const BufferCollection& data,
                            MatrixBufferFloat& impurityOut,
                            MatrixBufferFloat& thresholdOut,
                            MatrixBufferFloat& childCountsOut,
                            MatrixBufferFloat& leftYsOut,
                            MatrixBufferFloat& rightYsOut) const {}
};

