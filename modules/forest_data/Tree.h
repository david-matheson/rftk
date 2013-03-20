#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"

class ForestStats
{
public:
    ForestStats();
    void ProcessLeaf(int depth, int numberEstimatorPoints);
    float GetAverageDepth() const;
    float GetAverageEstimatorPoints() const;
    void Print() const;

    int mNumberOfLeafNodes;
    int mMinDepth;
    int mMaxDepth;
    int mTotalDepth;
    int mMinEstimatorPoints;
    int mMaxEstimatorPoints;
    int mTotalEstimatorPoints;
};

const int NULL_CHILD = -1;

class Tree
{
public:
    Tree();     //default for stl vector
    Tree(   const Int32MatrixBuffer& path,
            const Int32MatrixBuffer& intFeatureParams,
            const Float32MatrixBuffer& floatFeatureParams,
            const Int32VectorBuffer& depths,
            const Float32VectorBuffer& counts,
            const Float32MatrixBuffer& ys );
    Tree( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim );
    void GatherStats(ForestStats& stats) const;
    int NextNodeIndex();
    void Compact();

    Int32MatrixBuffer mPath;
    Int32MatrixBuffer mIntFeatureParams;
    Float32MatrixBuffer mFloatFeatureParams;
    Float32VectorBuffer mCounts;
    Int32VectorBuffer mDepths;
    Float32MatrixBuffer mYs;
private:
    int mLastNodeIndex;
    bool mValid;
};
