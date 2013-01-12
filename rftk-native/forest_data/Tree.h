#pragma once

#include "MatrixBuffer.h"

class Tree
{
public:
    Tree() {} //default for stl vector
    Tree(   const MatrixBufferInt& path,
            const MatrixBufferInt& intFeatureParams,
            const MatrixBufferFloat& floatFeatureParams,
            const MatrixBufferFloat& ys );

    MatrixBufferInt mPath;
    MatrixBufferInt mIntFeatureParams;
    MatrixBufferFloat mFloatFeatureParams;
    MatrixBufferFloat mYs;
};
