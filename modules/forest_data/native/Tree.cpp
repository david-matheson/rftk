#include <stdio.h>
#include <limits>

#include <asserts.h>
#include "Tree.h"
#include "ForestStats.h"

Tree::Tree()
: mPath(0,0, NULL_CHILD)
, mIntFeatureParams(0,0)
, mFloatFeatureParams(0,0)
, mCounts(0)
, mDepths(0)
, mYs(0,0)
, mExtraInfo()
, mLastNodeIndex(0)
, mValid(false)
{
}


Tree::Tree( const MatrixBufferTemplate<int>& path,
            const MatrixBufferTemplate<int>& intFeatureParams,
            const MatrixBufferTemplate<float>& floatFeatureParams,
            const VectorBufferTemplate<int> & depths,
            const VectorBufferTemplate<float>& counts,
            const MatrixBufferTemplate<float>& ys )
: mPath(path)
, mIntFeatureParams(intFeatureParams)
, mFloatFeatureParams(floatFeatureParams)
, mCounts(counts)
, mDepths(depths)
, mYs(ys)
, mExtraInfo()
, mLastNodeIndex(mPath.GetM())
, mValid(true)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mDepths.GetN())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mCounts.GetN())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())
}


Tree::Tree( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim  )
: mPath(initalNumberNodes, 2, NULL_CHILD)
, mIntFeatureParams(initalNumberNodes, maxIntParamsDim)
, mFloatFeatureParams(initalNumberNodes, maxFloatParamsDim)
, mCounts(initalNumberNodes)
, mDepths(initalNumberNodes)
, mYs(initalNumberNodes, maxYsDim)
, mExtraInfo()
, mLastNodeIndex(1)
, mValid(true)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())

    //TODO: This needs to be changed for regression
    mYs.SetAll(1.0f/static_cast<float>(mYs.GetN()));
}

Tree::Tree(const Tree& tree)
: mPath(tree.mPath)
, mIntFeatureParams(tree.mIntFeatureParams)
, mFloatFeatureParams(tree.mFloatFeatureParams)
, mCounts(tree.mCounts)
, mDepths(tree.mDepths)
, mYs(tree.mYs)
, mExtraInfo(tree.mExtraInfo)
, mLastNodeIndex(tree.mLastNodeIndex)
, mValid(tree.mValid)
{
}

Tree::~Tree()
{
}

void Tree::GatherStats(ForestStats& stats) const
{
    for(int nodeId=0; nodeId<mLastNodeIndex-1; nodeId++)
    {
        const bool isLeaf = (mPath.Get(nodeId, 0) == NULL_CHILD
                             || mPath.Get(nodeId, 1) == NULL_CHILD);
        if( isLeaf )
        {
            // printf("Tree::GatherStats %d %d %d\n", nodeId, depth, numberEstimatorSamples);
            stats.ProcessLeaf(*this, nodeId);
        }
    }
}

int Tree::NextNodeIndex()
{
    int nextNodeIndex = mLastNodeIndex;
    mLastNodeIndex++;
    const int numberOfNodesAllocated = mPath.GetM();
    if( mLastNodeIndex >= numberOfNodesAllocated )
    {
        int newNumberOfNodesAllocated = mLastNodeIndex + numberOfNodesAllocated/2 + 1;
        mPath.Resize(newNumberOfNodesAllocated, 2, NULL_CHILD);
        mIntFeatureParams.Resize(newNumberOfNodesAllocated, mIntFeatureParams.GetN());
        mFloatFeatureParams.Resize(newNumberOfNodesAllocated, mFloatFeatureParams.GetN());
        mCounts.Resize(newNumberOfNodesAllocated);
        mDepths.Resize(newNumberOfNodesAllocated);
        mYs.Resize(newNumberOfNodesAllocated, mYs.GetN());
    }
    return nextNodeIndex;
}

void Tree::Compact()
{
    mPath.Resize(mLastNodeIndex, 2, NULL_CHILD);
    mIntFeatureParams.Resize(mLastNodeIndex, mIntFeatureParams.GetN());
    mFloatFeatureParams.Resize(mLastNodeIndex, mFloatFeatureParams.GetN());
    mCounts.Resize(mLastNodeIndex);
    mDepths.Resize(mLastNodeIndex);
    mYs.Resize(mLastNodeIndex, mYs.GetN());
}