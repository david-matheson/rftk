#pragma once

#include <vector>
#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "ClassEntropyUtils.h"

// ----------------------------------------------------------------------------
//
// Compute the class information gain when walking sorted feature values.
// This class is called from BestSplitpointsWalkingSortedStep where
// MoveLeftToRight is called for the sorted feature values
//
// ----------------------------------------------------------------------------
template <class BT>
class ClassInfoGainWalker
{
public:
    ClassInfoGainWalker (const BufferId& sampleWeights,
                  const BufferId& classes,
                  const int numberOfClasses );
    virtual ~ClassInfoGainWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Reset();

    void MoveLeftToRight(typename BT::Index sampleIndex);

    typename BT::ImpurityValue Impurity();

    typename BT::Index GetYDim() const;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetLeftYs() const;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> GetRightYs() const;
    typename BT::SufficientStatsContinuous GetLeftChildCounts() const;
    typename BT::SufficientStatsContinuous GetRightChildCounts() const;

    typedef BT BufferTypes;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;

    VectorBufferTemplate<typename BT::ParamsContinuous> const* mSampleWeights;
    VectorBufferTemplate<typename BT::SourceInteger> const* mClasses;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mAllClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightClassHistogram;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftLogClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightLogClassHistogram;

    typename BT::SufficientStatsContinuous mStartEntropy;

    std::vector<bool> mRecomputeClassLog;
};


template <class BT>
ClassInfoGainWalker<BT>::ClassInfoGainWalker(const BufferId& sampleWeights,
                                                                      const BufferId& classes,
                                                                      const int numberOfClasses )
: mSampleWeightsBufferId(sampleWeights)
, mClassesBufferId(classes)
, mNumberOfClasses(numberOfClasses)
, mSampleWeights(NULL)
, mClasses(NULL)
, mAllClassHistogram(numberOfClasses)
, mLeftClassHistogram(numberOfClasses)
, mRightClassHistogram(numberOfClasses)
, mLeftLogClassHistogram(numberOfClasses)
, mRightLogClassHistogram(numberOfClasses)
, mStartEntropy(0)
, mRecomputeClassLog(numberOfClasses)
{}

template <class BT>
ClassInfoGainWalker<BT>::~ClassInfoGainWalker()
{}

template <class BT>
void ClassInfoGainWalker<BT>::Bind(const BufferCollectionStack& readCollection)
{
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<typename BT::ParamsContinuous> >(mSampleWeightsBufferId));
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<typename BT::SourceInteger> >(mClassesBufferId));
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::ParamsContinuous> >(mSampleWeightsBufferId);
    mClasses = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::SourceInteger> >(mClassesBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mClasses->GetN())

    for(typename BT::Index i=0; i<mSampleWeights->GetN(); i++)
    {
        mAllClassHistogram.Incr(mClasses->Get(i), mSampleWeights->Get(i));
    }

    Reset();
}

template <class BT>
void ClassInfoGainWalker<BT>::Reset()
{
    typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0.0);

    mLeftClassHistogram.Zero();
    mRightClassHistogram.Zero();
    mLeftLogClassHistogram.Zero();
    mRightLogClassHistogram.Zero();
    mRecomputeClassLog.clear();

    mLeftClassHistogram = mAllClassHistogram;

    for(typename BT::Index c=0; c<mNumberOfClasses; c++)
    {
        const typename BT::SufficientStatsContinuous logClass = 
                mAllClassHistogram.Get(c) > zero ? log2(mAllClassHistogram.Get(c)) : zero;
        mLeftLogClassHistogram.Set(c, logClass);
    }
    mStartEntropy = calcDiscreteEntropy<BT>(mLeftClassHistogram.Sum(), mLeftClassHistogram, mLeftLogClassHistogram);
}

template <class BT>
void ClassInfoGainWalker<BT>::MoveLeftToRight(typename BT::Index sampleIndex)
{
    const typename BT::ParamsContinuous weight = mSampleWeights->Get(sampleIndex);
    const typename BT::Index classIndex = mClasses->Get(sampleIndex);
    mLeftClassHistogram.Incr(classIndex, -weight);
    mRightClassHistogram.Incr(classIndex, weight);
    mRecomputeClassLog[classIndex] = true;
}

template <class BT>
typename BT::ImpurityValue ClassInfoGainWalker<BT>::Impurity()
{
    typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0.0);

    for(typename BT::Index c=0; c<mNumberOfClasses; c++)
    {
        if(mRecomputeClassLog[c])
        {
            const typename BT::SufficientStatsContinuous leftLogClass = mLeftClassHistogram.Get(c) > zero? log2(mLeftClassHistogram.Get(c)) : zero;
            mLeftLogClassHistogram.Set(c, leftLogClass);
            const typename BT::SufficientStatsContinuous rightLogClass = mRightClassHistogram.Get(c) > zero ? log2(mRightClassHistogram.Get(c)) : zero;
            mRightLogClassHistogram.Set(c, rightLogClass);
        }
    }

    const typename BT::SufficientStatsContinuous leftWeight = mLeftClassHistogram.Sum();
    const typename BT::SufficientStatsContinuous rightWeight = mRightClassHistogram.Sum();
    const typename BT::SufficientStatsContinuous totalWeight = leftWeight + rightWeight;

    const typename BT::SufficientStatsContinuous leftEntropy = calcDiscreteEntropy<BT>(leftWeight, mLeftClassHistogram, mLeftLogClassHistogram);
    const typename BT::SufficientStatsContinuous rightEntropy = calcDiscreteEntropy<BT>(rightWeight, mRightClassHistogram, mRightLogClassHistogram);

    const typename BT::ImpurityValue infoGain = mStartEntropy
                                                      - ((leftWeight / totalWeight) * leftEntropy)
                                                      - ((rightWeight / totalWeight) * rightEntropy);
    return infoGain;
}

template <class BT>
typename BT::Index ClassInfoGainWalker<BT>::GetYDim() const
{
    return mNumberOfClasses;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> ClassInfoGainWalker<BT>::GetLeftYs() const
{
    return mLeftClassHistogram.Normalized();
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> ClassInfoGainWalker<BT>::GetRightYs() const
{
    return mRightClassHistogram.Normalized();
}

template <class BT>
typename BT::SufficientStatsContinuous ClassInfoGainWalker<BT>::GetLeftChildCounts() const
{
    return mLeftClassHistogram.Sum();
}

template <class BT>
typename BT::SufficientStatsContinuous ClassInfoGainWalker<BT>::GetRightChildCounts() const
{
    return mRightClassHistogram.Sum();
}