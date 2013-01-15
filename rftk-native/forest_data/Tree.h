#pragma once

#include "MatrixBuffer.h"

class Tree
{
public:
    Tree()     //default for stl vector
    : mLastNodeIndex(0)
    , mValid(false) {}

    Tree(   const MatrixBufferInt& path,
            const MatrixBufferInt& intFeatureParams,
            const MatrixBufferFloat& floatFeatureParams,
            const MatrixBufferInt& depths,
            const MatrixBufferFloat& ys );
    Tree( int maxNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim );

    MatrixBufferInt mPath;
    MatrixBufferInt mIntFeatureParams;
    MatrixBufferFloat mFloatFeatureParams;
    MatrixBufferInt mDepths;
    MatrixBufferFloat mYs;
    int mLastNodeIndex;
    bool mValid;
};
