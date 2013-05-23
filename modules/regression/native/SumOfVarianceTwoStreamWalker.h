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
template <class FloatType, class IntType>
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

    void MoveLeftToRight(IntType sampleIndex);

    FloatType Impurity();

    IntType GetYDim() const;
    VectorBufferTemplate<FloatType> GetLeftEstimationYs() const;
    VectorBufferTemplate<FloatType> GetLeftStructureYs() const;
    VectorBufferTemplate<FloatType> GetRightEstimationYs() const;
    VectorBufferTemplate<FloatType> GetRightStructureYs() const;
    FloatType GetLeftEstimationChildCounts() const;
    FloatType GetLeftStructureChildCounts() const;
    FloatType GetRightEstimationChildCounts() const;
    FloatType GetRightStructureChildCounts() const;

    typedef FloatType Float;
    typedef IntType Int;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mStreamTypeBufferId;
    const BufferId mYsBufferId;
    const int mYdim;

    VectorBufferTemplate<FloatType> const* mSampleWeights;
    VectorBufferTemplate<IntType> const* mStreamType;
    MatrixBufferTemplate<FloatType> const* mYs;

    FloatType mStartEstimationCounts;
    FloatType mStartStructureCounts;
    VectorBufferTemplate<FloatType> mStartEstimationMeanVariance;
    VectorBufferTemplate<FloatType> mStartStructureMeanVariance;
    FloatType mLeftEstimationCounts;
    FloatType mLeftStructureCounts;
    VectorBufferTemplate<FloatType> mLeftEstimationMeanVariance;
    VectorBufferTemplate<FloatType> mLeftStructureMeanVariance;
    FloatType mRightEstimationCounts;
    FloatType mRightStructureCounts;
    VectorBufferTemplate<FloatType> mRightEstimationMeanVariance;
    VectorBufferTemplate<FloatType> mRightStructureMeanVariance;

    FloatType mStartVariance;
};


template <class FloatType, class IntType>
SumOfVarianceTwoStreamWalker<FloatType, IntType>::SumOfVarianceTwoStreamWalker(const BufferId& sampleWeights,
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

template <class FloatType, class IntType>
SumOfVarianceTwoStreamWalker<FloatType, IntType>::~SumOfVarianceTwoStreamWalker()
{}

template <class FloatType, class IntType>
void SumOfVarianceTwoStreamWalker<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection)
{
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);
    mYs = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mYsBufferId);
    mStreamType = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mStreamTypeBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mYs->GetM())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        const FloatType weight = mSampleWeights->Get(i);
        const IntType streamType = mStreamType->Get(i);

        VectorBufferTemplate<FloatType>& meanVariance = (streamType == STREAM_ESTIMATION) ?
                    mStartEstimationMeanVariance : mStartStructureMeanVariance;

        for(int d=0; d<mYdim; d++)
        {
            const FloatType y_d = mYs->Get(i,d);
            meanVariance.Incr(d, weight*y_d);
            meanVariance.Incr(d+mYdim, weight*y_d*y_d);
        }

        FloatType& counts = (streamType == STREAM_ESTIMATION) ? mStartEstimationCounts : mStartStructureCounts;
        counts += weight;
    }

    mStartVariance = FloatType(0);
    for(int d=0; d<mYdim; d++)
    {
        const FloatType y = mStartStructureMeanVariance.Get(d);
        const FloatType ySquared = mStartStructureMeanVariance.Get(d+mYdim);
        mStartVariance += ySquared / mStartStructureCounts - pow(y/mStartStructureCounts, 2);
    }

    Reset();
}

template <class FloatType, class IntType>
void SumOfVarianceTwoStreamWalker<FloatType, IntType>::Reset()
{
    mLeftEstimationCounts = mStartEstimationCounts;
    mLeftStructureCounts = mStartStructureCounts;
    mLeftEstimationMeanVariance = mStartEstimationMeanVariance;
    mLeftStructureMeanVariance = mStartStructureMeanVariance;
    mRightEstimationCounts = FloatType(0);
    mRightStructureCounts = FloatType(0);
    mRightEstimationMeanVariance.Zero();
    mRightStructureMeanVariance.Zero();
}

template <class FloatType, class IntType>
void SumOfVarianceTwoStreamWalker<FloatType, IntType>::MoveLeftToRight(IntType sampleIndex)
{
    const FloatType weight = mSampleWeights->Get(sampleIndex);
    const IntType streamType = mStreamType->Get(sampleIndex);

    VectorBufferTemplate<FloatType>& leftMeanVariance = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationMeanVariance : mLeftStructureMeanVariance;

    VectorBufferTemplate<FloatType>& rightMeanVariance = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationMeanVariance : mRightStructureMeanVariance;

    FloatType& leftCounts = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationCounts : mLeftStructureCounts;

    FloatType& rightCounts = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationCounts : mRightStructureCounts;

    leftCounts -= weight;
    rightCounts += weight;

    for(int d=0; d<mYdim; d++)
    {
        const FloatType y_d = mYs->Get(sampleIndex,d);
        leftMeanVariance.Incr(d, -weight*y_d);
        leftMeanVariance.Incr(d+mYdim, -weight*y_d*y_d);
        rightMeanVariance.Incr(d, weight*y_d);
        rightMeanVariance.Incr(d+mYdim, weight*y_d*y_d);
    }
}

template <class FloatType, class IntType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType>::Impurity()
{
    const FloatType countsTotal = mLeftStructureCounts+mRightStructureCounts;
    FloatType leftSumOfVariance = FloatType(0);
    FloatType rightSumOfVariance = FloatType(0);

    for(int d=0; d<mYdim; d++)
    {
        const FloatType leftY = mLeftStructureMeanVariance.Get(d);
        const FloatType leftYSquared = mLeftStructureMeanVariance.Get(d+mYdim);
        const FloatType rightY = mRightStructureMeanVariance.Get(d);
        const FloatType rightYSquared = mRightStructureMeanVariance.Get(d+mYdim);

        leftSumOfVariance += (mLeftStructureCounts>0.0) ? leftYSquared/mLeftStructureCounts -  pow(leftY/mLeftStructureCounts, 2) : 0.0;
        rightSumOfVariance += (mRightStructureCounts>0.0) ? rightYSquared/mRightStructureCounts -  pow(rightY/mRightStructureCounts, 2) : 0.0;
    }

    const FloatType varianceGain = mStartVariance
                                  - ((mLeftStructureCounts / countsTotal) * leftSumOfVariance)
                                  - ((mRightStructureCounts / countsTotal) * rightSumOfVariance);

    return varianceGain;
}

template <class FloatType, class IntType>
IntType SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetYDim() const
{
    return mYdim*2;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetLeftEstimationYs() const
{
    return mLeftEstimationMeanVariance;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetLeftStructureYs() const
{
    return mLeftStructureMeanVariance;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetRightEstimationYs() const
{
    return mRightEstimationMeanVariance;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetRightStructureYs() const
{
    return mRightStructureMeanVariance;
}


template <class FloatType, class IntType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetLeftEstimationChildCounts() const
{
    return mLeftEstimationCounts;
}

template <class FloatType, class IntType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetLeftStructureChildCounts() const
{
    return mLeftStructureCounts;
}

template <class FloatType, class IntType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetRightEstimationChildCounts() const
{
    return mRightEstimationCounts;
}

template <class FloatType, class IntType>
FloatType SumOfVarianceTwoStreamWalker<FloatType, IntType>::GetRightStructureChildCounts() const
{
    return mRightStructureCounts;
}