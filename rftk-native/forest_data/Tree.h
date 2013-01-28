#pragma once

#include "MatrixBuffer.h"

class Tree
{
public:
    Tree()     //default for stl vector
    : mLastNodeIndex(0)
    , mValid(false) {}

    Tree(   const Int32MatrixBuffer& path,
            const Int32MatrixBuffer& intFeatureParams,
            const Float32MatrixBuffer& floatFeatureParams,
            const Int32MatrixBuffer& depths,
            const Float32MatrixBuffer& ys );
    Tree( int maxNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim );

    Int32MatrixBuffer mPath;
    Int32MatrixBuffer mIntFeatureParams;
    Float32MatrixBuffer mFloatFeatureParams;
    Int32MatrixBuffer mDepths;
    Float32MatrixBuffer mYs;
    int mLastNodeIndex;
    bool mValid;
};
