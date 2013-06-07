#pragma once

#include <VectorBuffer.h>
#include <MatrixBuffer.h>

const int NULL_CHILD = -1;

class ForestStats;

class Tree
{
public:
    Tree();     //default for stl vector
    Tree(   const MatrixBufferTemplate<int>& path,
            const MatrixBufferTemplate<int>& intFeatureParams,
            const MatrixBufferTemplate<float>& floatFeatureParams,
            const VectorBufferTemplate<int> & depths,
            const VectorBufferTemplate<float>& counts,
            const MatrixBufferTemplate<float>& ys );
    Tree( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim );
    void GatherStats(ForestStats& stats) const;
    int NextNodeIndex();
    void Compact();

    MatrixBufferTemplate<int> mPath;
    MatrixBufferTemplate<int> mIntFeatureParams;
    MatrixBufferTemplate<float> mFloatFeatureParams;
    VectorBufferTemplate<float> mCounts;
    VectorBufferTemplate<int> mDepths;
    MatrixBufferTemplate<float> mYs;
private:
    int mLastNodeIndex;
    bool mValid;
};


