#include <stdio.h>
#include <limits>

#include "assert_util.h"
#include "Tree.h"

ForestStats::ForestStats()
: mNumberOfLeafNodes(0)
, mMinDepth(std::numeric_limits<int>::max())
, mMaxDepth(0)
, mTotalDepth(0)
, mMinEstimatorPoints(std::numeric_limits<int>::max())
, mMaxEstimatorPoints(0)
, mTotalEstimatorPoints(0)
{}

void ForestStats::ProcessLeaf(int depth, int numberEstimatorPoints)
{
    mNumberOfLeafNodes++;
    mMinDepth = std::min(mMinDepth, depth);
    mMaxDepth = std::max(mMaxDepth, depth);
    mTotalDepth += depth;
    mMinEstimatorPoints = std::min(mMinEstimatorPoints, numberEstimatorPoints);
    mMaxEstimatorPoints = std::max(mMaxEstimatorPoints, numberEstimatorPoints);
    mTotalEstimatorPoints += numberEstimatorPoints;
}

float ForestStats::GetAverageDepth() const
{
    return static_cast<float>(mTotalDepth) / static_cast<float>(mNumberOfLeafNodes);
}

float ForestStats::GetAverageEstimatorPoints() const
{
    return static_cast<float>(mTotalEstimatorPoints) / static_cast<float>(mNumberOfLeafNodes);
}

void ForestStats::Print() const
{
    printf("ForestStats: #leafs=%d min-depth=%d max-depth=%d avg-depth=%0.2f min-points=%d max-points=%d avg-points=%0.2f\n",
        mNumberOfLeafNodes, mMinDepth, mMaxDepth, GetAverageDepth(),
        mMinEstimatorPoints, mMaxEstimatorPoints, GetAverageEstimatorPoints());
}

Tree::Tree()
: mPath(0,0, NULL_CHILD)
, mIntFeatureParams(0,0)
, mFloatFeatureParams(0,0)
, mCounts(0)
, mDepths(0)
, mYs(0,0)
, mLastNodeIndex(0)
, mValid(false)
{}

Tree::Tree( const Int32MatrixBuffer& path,
            const Int32MatrixBuffer& intFeatureParams,
            const Float32MatrixBuffer& floatFeatureParams,
            const Int32VectorBuffer& depths,
            const Float32VectorBuffer& counts,
            const Float32MatrixBuffer& ys )
: mPath(path)
, mIntFeatureParams(intFeatureParams)
, mFloatFeatureParams(floatFeatureParams)
, mCounts(counts)
, mDepths(depths)
, mYs(ys)
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
, mLastNodeIndex(1)
, mValid(true)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())

    //TODO: This needs to be changed for regression
    mYs.SetAll(1.0f/static_cast<float>(mYs.GetN()));
}

void Tree::GatherStats(ForestStats& stats) const
{
    for(int nodeId=0; nodeId<mLastNodeIndex-1; nodeId++)
    {
        const bool isLeaf = (mPath.Get(nodeId, 0) == NULL_CHILD
                             || mPath.Get(nodeId, 1) == NULL_CHILD);
        if( isLeaf )
        {
            const int depth = mDepths.Get(nodeId);
            const int numberEstimatorSamples = static_cast<int>(mCounts.Get(nodeId));
            // printf("Tree::GatherStats %d %d %d\n", nodeId, depth, numberEstimatorSamples);
            stats.ProcessLeaf(depth, numberEstimatorSamples);
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