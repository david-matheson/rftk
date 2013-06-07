#include <stdio.h>
#include "ForestStats.h"

ForestStats::ForestStats()
: mNumberOfLeafNodes(0)
, mMinDepth(std::numeric_limits<int>::max())
, mMaxDepth(0)
, mTotalDepth(0)
, mMinEstimatorPoints(std::numeric_limits<int>::max())
, mMaxEstimatorPoints(0)
, mTotalEstimatorPoints(0)
, mMinError(1.0f)
, mMaxError(0.0f)
, mTotalError(0.0f)
{}

void ForestStats::ProcessLeaf(const Tree& tree, int nodeId)
{
    const int depth = tree.mDepths.Get(nodeId);
    mNumberOfLeafNodes++;
    mMinDepth = std::min(mMinDepth, depth);
    mMaxDepth = std::max(mMaxDepth, depth);
    mTotalDepth += depth;

    const int numberEstimatorPoints = static_cast<int>(tree.mCounts.Get(nodeId));
    mMinEstimatorPoints = std::min(mMinEstimatorPoints, numberEstimatorPoints);
    mMaxEstimatorPoints = std::max(mMaxEstimatorPoints, numberEstimatorPoints);
    mTotalEstimatorPoints += numberEstimatorPoints;

    const float errorProb = 1.0f - tree.mYs.SliceRow(nodeId).GetMax();
    mMinError = std::min(mMinError, errorProb);
    mMaxError = std::max(mMaxError, errorProb);
    mTotalError += errorProb;
}

float ForestStats::GetAverageDepth() const
{
    return static_cast<float>(mTotalDepth) / static_cast<float>(mNumberOfLeafNodes);
}

float ForestStats::GetAverageEstimatorPoints() const
{
    return static_cast<float>(mTotalEstimatorPoints) / static_cast<float>(mNumberOfLeafNodes);
}

float ForestStats::GetAverageError() const
{
    return mTotalError / static_cast<float>(mNumberOfLeafNodes);
}

void ForestStats::Print() const
{
    printf("ForestStats: #leafs=%d min-depth=%d max-depth=%d avg-depth=%0.2f min-points=%d max-points=%d avg-points=%0.2f min-error=%0.2f max-error=%0.2f avg-error=%0.2f\n",
        mNumberOfLeafNodes, mMinDepth, mMaxDepth, GetAverageDepth(),
        mMinEstimatorPoints, mMaxEstimatorPoints, GetAverageEstimatorPoints(),
        mMinError, mMaxError, GetAverageError());
}