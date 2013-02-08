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

Tree::Tree( const Int32MatrixBuffer& path,
            const Int32MatrixBuffer& intFeatureParams,
            const Float32MatrixBuffer& floatFeatureParams,
            const Int32VectorBuffer& depths,
            const Float32VectorBuffer& counts,
            const Float32MatrixBuffer& ys )
: mPath(path)
, mIntFeatureParams(intFeatureParams)
, mFloatFeatureParams(floatFeatureParams)
, mDepths(depths)
, mCounts(counts)
, mYs(ys)
, mValid(true)
, mLastNodeIndex(1)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mDepths.GetN())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mCounts.GetN())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())
}

Tree::Tree( int maxNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim  )
: mPath(maxNumberNodes, 2)
, mIntFeatureParams(maxNumberNodes, maxIntParamsDim)
, mFloatFeatureParams(maxNumberNodes, maxFloatParamsDim)
, mDepths(maxNumberNodes)
, mCounts(maxNumberNodes)
, mYs(maxNumberNodes, maxYsDim)
, mValid(true)
, mLastNodeIndex(1)
{
    ASSERT_ARG_DIM_1D(mPath.GetM(), mIntFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mFloatFeatureParams.GetM())
    ASSERT_ARG_DIM_1D(mPath.GetM(), mYs.GetM())

    //TODO: This needs to be changed for regression
    mYs.SetAll(1.0f/static_cast<float>(mYs.GetN()));

    mPath.SetAll(-1);
}

void Tree::GatherStats(ForestStats& stats) const
{
    for(int nodeId=0; nodeId<mLastNodeIndex; nodeId++)
    {
        const bool isLeaf = (mPath.Get(nodeId, 0) == -1
                             && mPath.Get(nodeId, 1) == -1);
        if( isLeaf )
        {
            const int depth = mDepths.Get(nodeId);
            const int numberEstimatorSamples = static_cast<int>(mCounts.Get(nodeId));
            stats.ProcessLeaf(depth, numberEstimatorSamples);
        }
    }
}