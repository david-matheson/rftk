#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "FeatureExtractorStep.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

template <class FloatType, class IntType>
class TestBufferWalker
{
public:
    TestBufferWalker (const MatrixBufferTemplate<FloatType>& impurities,
                  const MatrixBufferTemplate<FloatType>& leftYs,
                  const MatrixBufferTemplate<FloatType>& rightYs);
    virtual ~TestBufferWalker();

    void Bind(const BufferCollectionStack& readCollection);
    void Reset();

    void MoveLeftToRight(IntType sampleIndex);

    FloatType Impurity() const;

    IntType GetYDim() const;
    VectorBufferTemplate<FloatType> GetLeftYs() const;
    VectorBufferTemplate<FloatType> GetRightYs() const;
    FloatType GetLeftChildCounts() const;
    FloatType GetRightChildCounts() const;

    typedef FloatType Float;
    typedef IntType Int;

private:
    const MatrixBufferTemplate<FloatType> mImpurities;
    const MatrixBufferTemplate<FloatType> mLeftYs;
    const MatrixBufferTemplate<FloatType> mRightYs;
    int mFeatureIndex;
    int mSampleIndex;
};


template <class FloatType, class IntType>
TestBufferWalker<FloatType, IntType>::TestBufferWalker(const MatrixBufferTemplate<FloatType>& impurities,
                                              const MatrixBufferTemplate<FloatType>& leftYs,
                                              const MatrixBufferTemplate<FloatType>& rightYs )
: mImpurities(impurities)
, mLeftYs(leftYs)
, mRightYs(rightYs)
, mFeatureIndex(-1) //Reset will be called before ProcessNextSample
, mSampleIndex(0)
{}

template <class FloatType, class IntType>
TestBufferWalker<FloatType, IntType>::~TestBufferWalker()
{}

template <class FloatType, class IntType>
void TestBufferWalker<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection)
{
}

template <class FloatType, class IntType>
void TestBufferWalker<FloatType, IntType>::Reset()
{
    mFeatureIndex++;
}

template <class FloatType, class IntType>
void TestBufferWalker<FloatType, IntType>::MoveLeftToRight(IntType sampleIndex)
{
    mSampleIndex = sampleIndex;
}

template <class FloatType, class IntType>
FloatType TestBufferWalker<FloatType, IntType>::Impurity() const
{
    const FloatType infoGain = mImpurities.Get(mFeatureIndex, mSampleIndex);
    return infoGain;
}

template <class FloatType, class IntType>
IntType TestBufferWalker<FloatType, IntType>::GetYDim() const
{
    return mLeftYs.GetN();
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> TestBufferWalker<FloatType, IntType>::GetLeftYs() const
{
    return mLeftYs.SliceRowAsVector(mFeatureIndex);
}

template <class FloatType, class IntType>
VectorBufferTemplate<FloatType> TestBufferWalker<FloatType, IntType>::GetRightYs() const
{
    return mRightYs.SliceRowAsVector(mFeatureIndex);
}

template <class FloatType, class IntType>
FloatType TestBufferWalker<FloatType, IntType>::GetLeftChildCounts() const
{
    return mLeftYs.SumRow(mFeatureIndex);
}

template <class FloatType, class IntType>
FloatType TestBufferWalker<FloatType, IntType>::GetRightChildCounts() const
{
    return mRightYs.SumRow(mFeatureIndex);
}