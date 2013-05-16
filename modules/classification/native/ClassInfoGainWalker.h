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
template <class FloatType, class IntType>
class ClassInfoGainWalker
{
public:
    ClassInfoGainWalker (const BufferId& sampleWeights,
                  const BufferId& classes,
                  const int numberOfClasses );
    virtual ~ClassInfoGainWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Reset();

    void MoveLeftToRight(IntType sampleIndex);

    FloatType Impurity();

    IntType GetYDim() const;
    VectorBufferTemplate<FloatType> GetLeftYs() const;
    VectorBufferTemplate<FloatType> GetRightYs() const;
    FloatType GetLeftChildCounts() const;
    FloatType GetRightChildCounts() const;

    typedef FloatType Float;
    typedef IntType Int;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;

    VectorBufferTemplate<FloatType> const* mSampleWeights;
    VectorBufferTemplate<IntType> const* mClasses;

    VectorBufferTemplate<FloatType> mAllClassHistogram;
    VectorBufferTemplate<FloatType> mLeftClassHistogram;
    VectorBufferTemplate<FloatType> mRightClassHistogram;

    VectorBufferTemplate<FloatType> mLeftLogClassHistogram;
    VectorBufferTemplate<FloatType> mRightLogClassHistogram;

    FloatType mStartEntropy;

    std::vector<bool> mRecomputeClassLog;
};


template <class FloatType, class IntType>
ClassInfoGainWalker<FloatType, IntType>::ClassInfoGainWalker(const BufferId& sampleWeights,
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

template <class FloatType, class IntType>
ClassInfoGainWalker<FloatType, IntType>::~ClassInfoGainWalker()
{}

template <class FloatType, class IntType>
void ClassInfoGainWalker<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection)
{
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId));
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<IntType> >(mClassesBufferId));
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);
    mClasses = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mClassesBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mClasses->GetN())

    for(int i=0; i<mSampleWeights->GetN(); i++)
    {
        mAllClassHistogram.Incr(mClasses->Get(i), mSampleWeights->Get(i));
    }

    Reset();
}

template <class FloatType, class IntType>
void ClassInfoGainWalker<FloatType, IntType>::Reset()
{
    mLeftClassHistogram.Zero();
    mRightClassHistogram.Zero();
    mLeftLogClassHistogram.Zero();
    mRightLogClassHistogram.Zero();
    mRecomputeClassLog.clear();

    mLeftClassHistogram = mAllClassHistogram;

    for(int c=0; c<mNumberOfClasses; c++)
    {
        const FloatType logClass = mAllClassHistogram.Get(c) > 0.0f ? log2(mAllClassHistogram.Get(c)) : 0.0f;
        mLeftLogClassHistogram.Set(c, logClass);
        mStartEntropy = calcDiscreteEntropy<FloatType>(mLeftClassHistogram.Sum(), mLeftClassHistogram, mLeftLogClassHistogram);
    }
}

template <class FloatType, class IntType>
void ClassInfoGainWalker<FloatType, IntType>::MoveLeftToRight(IntType sampleIndex)
{
    const FloatType weight = mSampleWeights->Get(sampleIndex);
    const IntType classIndex = mClasses->Get(sampleIndex);
    mLeftClassHistogram.Incr(classIndex, -weight);
    mRightClassHistogram.Incr(classIndex, weight);
    mRecomputeClassLog[classIndex] = true;
}

template <class FloatType, class IntType>
FloatType ClassInfoGainWalker<FloatType, IntType>::Impurity()
{
    for(int c=0; c<mNumberOfClasses; c++)
    {
        if(mRecomputeClassLog[c])
        {
            const FloatType leftLogClass = mLeftClassHistogram.Get(c) > 0.0f ? log2(mLeftClassHistogram.Get(c)) : 0.0f;
            mLeftLogClassHistogram.Set(c, leftLogClass);
            const FloatType rightLogClass = mRightClassHistogram.Get(c) > 0.0f ? log2(mRightClassHistogram.Get(c)) : 0.0f;
            mRightLogClassHistogram.Set(c, rightLogClass);
        }
    }

    const FloatType leftWeight = mLeftClassHistogram.Sum();
    const FloatType rightWeight = mRightClassHistogram.Sum();
    const FloatType totalWeight = leftWeight + rightWeight;

    const FloatType leftEntropy = calcDiscreteEntropy<FloatType>(leftWeight, mLeftClassHistogram, mLeftLogClassHistogram);
    const FloatType rightEntropy = calcDiscreteEntropy<FloatType>(rightWeight, mRightClassHistogram, mRightLogClassHistogram);

    const FloatType infoGain = mStartEntropy
                                  - ((leftWeight / totalWeight) * leftEntropy)
                                  - ((rightWeight / totalWeight) * rightEntropy);
    return infoGain;
}

template <class FloatType, class IntType>
IntType ClassInfoGainWalker<FloatType, IntType>::GetYDim() const
{
    return mNumberOfClasses;
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> ClassInfoGainWalker<FloatType, IntType>::GetLeftYs() const
{
    return mLeftClassHistogram.Normalized();
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> ClassInfoGainWalker<FloatType, IntType>::GetRightYs() const
{
    return mRightClassHistogram.Normalized();
}

template <class FloatType, class IntType>
FloatType ClassInfoGainWalker<FloatType, IntType>::GetLeftChildCounts() const
{
    return mLeftClassHistogram.Sum();
}

template <class FloatType, class IntType>
FloatType ClassInfoGainWalker<FloatType, IntType>::GetRightChildCounts() const
{
    return mRightClassHistogram.Sum();
}