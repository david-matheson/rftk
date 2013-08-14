#pragma once

#include <vector>
#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "AssignStreamStep.h"

// ----------------------------------------------------------------------------
//
// Compute the sum of the variance for each component
// This class is called from BestSplitpointsWalkingSortedStep where
// MoveLeftToRight is called for the sorted feature values
//
// ----------------------------------------------------------------------------
template <class BT>
class SumOfVarianceTwoStreamWalker
{
public:
    SumOfVarianceTwoStreamWalker (const BufferId& sampleWeights,
                                    const BufferId& streamType,
                                    const BufferId& ys,
                                    const int ydim );
    virtual ~SumOfVarianceTwoStreamWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Reset();

    void MoveLeftToRight(typename BT::Index sampleIndex);

    typename BT::ImpurityValue Impurity();

    typename BT::Index GetYDim() const;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetLeftEstimationYs() const;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetLeftStructureYs() const;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetRightEstimationYs() const;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetRightStructureYs() const;
    typename BT::SufficientStatsContinuous GetLeftEstimationChildCounts() const;
    typename BT::SufficientStatsContinuous GetLeftStructureChildCounts() const;
    typename BT::SufficientStatsContinuous GetRightEstimationChildCounts() const;
    typename BT::SufficientStatsContinuous GetRightStructureChildCounts() const;

    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mStreamTypeBufferId;
    const BufferId mYsBufferId;
    const int mYdim;

    VectorBufferTemplate<typename BT::DatapointCounts> const* mSampleWeights;
    VectorBufferTemplate<typename BT::ParamsInteger> const* mStreamType;
    MatrixBufferTemplate<typename BT::SourceContinuous> const* mYs;

    typename BT::SufficientStatsContinuous mStartEstimationCounts;
    typename BT::SufficientStatsContinuous mStartStructureCounts;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mStartEstimationMeanVariance;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mStartStructureMeanVariance;
    typename BT::SufficientStatsContinuous mLeftEstimationCounts;
    typename BT::SufficientStatsContinuous mLeftStructureCounts;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftEstimationMeanVariance;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftStructureMeanVariance;
    typename BT::SufficientStatsContinuous mRightEstimationCounts;
    typename BT::SufficientStatsContinuous mRightStructureCounts;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightEstimationMeanVariance;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightStructureMeanVariance;

    typename BT::SufficientStatsContinuous mStartVariance;
};


template <class BT>
SumOfVarianceTwoStreamWalker<BT>::SumOfVarianceTwoStreamWalker(const BufferId& sampleWeights,
                                                                      const BufferId& streamType,
                                                                      const BufferId& ys,
                                                                      const int ydim )
: mSampleWeightsBufferId(sampleWeights)
, mStreamTypeBufferId(streamType)
, mYsBufferId(ys)
, mYdim(ydim)
, mSampleWeights(NULL)
, mStreamType(NULL)
, mYs(NULL)
, mStartEstimationCounts(0)
, mStartStructureCounts(0)
, mStartEstimationMeanVariance(ydim*2)
, mStartStructureMeanVariance(ydim*2)
, mLeftEstimationCounts(0)
, mLeftStructureCounts(0)
, mLeftEstimationMeanVariance(ydim*2)
, mLeftStructureMeanVariance(ydim*2)
, mRightEstimationCounts(0)
, mRightStructureCounts(0)
, mRightEstimationMeanVariance(ydim*2)
, mRightStructureMeanVariance(ydim*2)
, mStartVariance(0)
{}

template <class BT>
SumOfVarianceTwoStreamWalker<BT>::~SumOfVarianceTwoStreamWalker()
{}

template <class BT>
void SumOfVarianceTwoStreamWalker<BT>::Bind(const BufferCollectionStack& readCollection)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::DatapointCounts> >(mSampleWeightsBufferId);
    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<typename BT::SourceContinuous> >(mYsBufferId);
    mStreamType = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::ParamsInteger> >(mStreamTypeBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        const typename BT::SufficientStatsContinuous weight = mSampleWeights->Get(i);
        const typename BT::ParamsInteger streamType = mStreamType->Get(i);

        VectorBufferTemplate<typename BT::SufficientStatsContinuous> meanVariance = (streamType == STREAM_ESTIMATION) ?
                    mStartEstimationMeanVariance : mStartStructureMeanVariance;

        typename BT::SufficientStatsContinuous& counts = (streamType == STREAM_ESTIMATION) ? mStartEstimationCounts : mStartStructureCounts;
        typename BT::SufficientStatsContinuous newCounts = counts + weight;

        for(int d=0; d<mYdim; d++)
        {
            const typename BT::SufficientStatsContinuous y_i = mYs->Get(i,d);
            // old unstable sufficient stats 
            // meanVariance.Incr(d, weight*y_d);
            // meanVariance.Incr(d+mYdim, weight*y_d*y_d);
            const typename BT::SufficientStatsContinuous mean = meanVariance.Get(d);
            const typename BT::SufficientStatsContinuous delta = y_i - mean;
            const typename BT::SufficientStatsContinuous r = delta * weight / newCounts;
            meanVariance.Incr(d,r);
            meanVariance.Incr(d+mYdim, counts*delta*r);
        }
        counts = newCounts;
    }

    mStartVariance = typename BT::SufficientStatsContinuous(0);
    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const FloatType y = mStartStructureMeanVariance.Get(d);
        // const FloatType ySquared = mStartStructureMeanVariance.Get(d+mYdim);
        // mStartVariance += ySquared / mStartStructureCounts - pow(y/mStartStructureCounts, 2);
        mStartVariance += mStartStructureMeanVariance.Get(d+mYdim) / mStartStructureCounts;
    }

    Reset();
}

template <class BT>
void SumOfVarianceTwoStreamWalker<BT>::Reset()
{
    mLeftEstimationCounts = mStartEstimationCounts;
    mLeftStructureCounts = mStartStructureCounts;
    mLeftEstimationMeanVariance = mStartEstimationMeanVariance;
    mLeftStructureMeanVariance = mStartStructureMeanVariance;
    mRightEstimationCounts = typename BT::SufficientStatsContinuous(0);
    mRightStructureCounts = typename BT::SufficientStatsContinuous(0);
    mRightEstimationMeanVariance.Zero();
    mRightStructureMeanVariance.Zero();
}

template <class BT>
void SumOfVarianceTwoStreamWalker<BT>::MoveLeftToRight(typename BT::Index sampleIndex)
{
    const typename BT::SufficientStatsContinuous weight = mSampleWeights->Get(sampleIndex);
    const typename BT::ParamsInteger streamType = mStreamType->Get(sampleIndex);

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> leftMeanVariance = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationMeanVariance : mLeftStructureMeanVariance;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> rightMeanVariance = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationMeanVariance : mRightStructureMeanVariance;

    typename BT::SufficientStatsContinuous& leftCounts = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationCounts : mLeftStructureCounts;

    typename BT::SufficientStatsContinuous& rightCounts = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationCounts : mRightStructureCounts;

    const typename BT::SufficientStatsContinuous newLeftCounts = leftCounts - weight;
    const typename BT::SufficientStatsContinuous newRightCounts = rightCounts + weight;

    
    for(int d=0; d<mYdim; d++)
    {
        const typename BT::SufficientStatsContinuous y_d = mYs->Get(sampleIndex,d);

        // old unstable sufficient stats 
        // leftMeanVariance.Incr(d, -weight*y_d);
        // leftMeanVariance.Incr(d+mYdim, -weight*y_d*y_d);
        // rightMeanVariance.Incr(d, weight*y_d);
        // rightMeanVariance.Incr(d+mYdim, weight*y_d*y_d);

        const typename BT::SufficientStatsContinuous leftMean = leftMeanVariance.Get(d);
        const typename BT::SufficientStatsContinuous leftDelta = y_d - leftMean;
        const typename BT::SufficientStatsContinuous leftr = leftDelta * -1.0 * weight / newLeftCounts;
        leftMeanVariance.Incr(d, leftr);
        leftMeanVariance.Incr(d+mYdim, leftCounts*leftDelta*leftr);

        const typename BT::SufficientStatsContinuous rightMean = rightMeanVariance.Get(d);
        const typename BT::SufficientStatsContinuous rightDelta = y_d - rightMean;
        const typename BT::SufficientStatsContinuous rightr = rightDelta * weight / newRightCounts;
        rightMeanVariance.Incr(d, rightr);
        rightMeanVariance.Incr(d+mYdim, rightCounts*rightDelta*rightr);
    }
    leftCounts = newLeftCounts;
    rightCounts = newRightCounts;
}

template <class BT>
typename BT::ImpurityValue SumOfVarianceTwoStreamWalker<BT>::Impurity()
{
    const typename BT::SufficientStatsContinuous countsTotal = mLeftStructureCounts+mRightStructureCounts;
    typename BT::SufficientStatsContinuous leftSumOfVariance = typename BT::SufficientStatsContinuous(0);
    typename BT::SufficientStatsContinuous rightSumOfVariance = typename BT::SufficientStatsContinuous(0);

    for(int d=0; d<mYdim; d++)
    {
        // old unstable sufficient stats 
        // const FloatType leftY = mLeftStructureMeanVariance.Get(d);
        // const FloatType leftYSquared = mLeftStructureMeanVariance.Get(d+mYdim);
        // const FloatType rightY = mRightStructureMeanVariance.Get(d);
        // const FloatType rightYSquared = mRightStructureMeanVariance.Get(d+mYdim);

        // leftSumOfVariance += (mLeftStructureCounts>0.0) ? leftYSquared/mLeftStructureCounts -  pow(leftY/mLeftStructureCounts, 2) : 0.0;
        // rightSumOfVariance += (mRightStructureCounts>0.0) ? rightYSquared/mRightStructureCounts -  pow(rightY/mRightStructureCounts, 2) : 0.0;

        const typename BT::SufficientStatsContinuous leftY2 = mLeftStructureMeanVariance.Get(d+mYdim);
        const typename BT::SufficientStatsContinuous rightY2 = mRightStructureMeanVariance.Get(d+mYdim);

        leftSumOfVariance += (mLeftStructureCounts>0.0) ? leftY2/mLeftStructureCounts : typename BT::SufficientStatsContinuous(0);
        rightSumOfVariance += (mRightStructureCounts>0.0) ? rightY2/mRightStructureCounts : typename BT::SufficientStatsContinuous(0);
    }

    const typename BT::ImpurityValue varianceGain = mStartVariance
                                  - ((mLeftStructureCounts / countsTotal) * leftSumOfVariance)
                                  - ((mRightStructureCounts / countsTotal) * rightSumOfVariance);

    return varianceGain;
}

template <class BT>
typename BT::Index SumOfVarianceTwoStreamWalker<BT>::GetYDim() const
{
    return mYdim*2;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> SumOfVarianceTwoStreamWalker<BT>::GetLeftEstimationYs() const
{
    return mLeftEstimationMeanVariance;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> SumOfVarianceTwoStreamWalker<BT>::GetLeftStructureYs() const
{
    return mLeftStructureMeanVariance;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> SumOfVarianceTwoStreamWalker<BT>::GetRightEstimationYs() const
{
    return mRightEstimationMeanVariance;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> SumOfVarianceTwoStreamWalker<BT>::GetRightStructureYs() const
{
    return mRightStructureMeanVariance;
}


template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceTwoStreamWalker<BT>::GetLeftEstimationChildCounts() const
{
    return mLeftEstimationCounts;
}

template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceTwoStreamWalker<BT>::GetLeftStructureChildCounts() const
{
    return mLeftStructureCounts;
}

template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceTwoStreamWalker<BT>::GetRightEstimationChildCounts() const
{
    return mRightEstimationCounts;
}

template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceTwoStreamWalker<BT>::GetRightStructureChildCounts() const
{
    return mRightStructureCounts;
}