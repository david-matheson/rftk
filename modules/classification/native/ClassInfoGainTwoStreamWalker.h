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
#include "ClassEntropyUtils.h"

// ----------------------------------------------------------------------------
//
// Compute the class information gain when walking sorted feature values.
// This class is called from BestSplitpointsWalkingSortedStep where
// MoveLeftToRight is called for the sorted feature values
//
// ----------------------------------------------------------------------------
template <class BT>
class ClassInfoGainTwoStreamWalker
{
public:
    ClassInfoGainTwoStreamWalker (const BufferId& sampleWeights,
                                const BufferId& streamType,
                                const BufferId& classes,
                                const int numberOfClasses );
    virtual ~ClassInfoGainTwoStreamWalker();

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
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;

    VectorBufferTemplate<typename BT::ParamsContinuous> const* mSampleWeights;
    VectorBufferTemplate<typename BT::ParamsInteger> const* mStreamType;
    VectorBufferTemplate<typename BT::SourceInteger> const* mClasses;


    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mStartEstimationClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mStartStructureClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mStartLogStructureClassHistogram;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftEstimationClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftStructureClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mLeftLogStructureClassHistogram;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightEstimationClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightStructureClassHistogram;
    VectorBufferTemplate<typename BT::SufficientStatsContinuous> mRightLogStructureClassHistogram;

    typename BT::SufficientStatsContinuous mStartEntropy;
    std::vector<bool> mRecomputeClassLog;
};


template <class BT>
ClassInfoGainTwoStreamWalker<BT>::ClassInfoGainTwoStreamWalker(const BufferId& sampleWeights,
                                                        const BufferId& streamType,
                                                        const BufferId& classes,
                                                        const int numberOfClasses )
: mSampleWeightsBufferId(sampleWeights)
, mStreamTypeBufferId(streamType)
, mClassesBufferId(classes)
, mNumberOfClasses(numberOfClasses)
, mSampleWeights(NULL)
, mStreamType(NULL)
, mClasses(NULL)
, mStartEstimationClassHistogram(numberOfClasses)
, mStartStructureClassHistogram(numberOfClasses)
, mStartLogStructureClassHistogram(numberOfClasses)
, mLeftEstimationClassHistogram(numberOfClasses)
, mLeftStructureClassHistogram(numberOfClasses)
, mLeftLogStructureClassHistogram(numberOfClasses)
, mRightEstimationClassHistogram(numberOfClasses)
, mRightStructureClassHistogram(numberOfClasses)
, mRightLogStructureClassHistogram(numberOfClasses)
, mStartEntropy(0)
, mRecomputeClassLog(numberOfClasses)
{}

template <class BT>
ClassInfoGainTwoStreamWalker<BT>::~ClassInfoGainTwoStreamWalker()
{}

template <class BT>
void ClassInfoGainTwoStreamWalker<BT>::Bind(const BufferCollectionStack& readCollection)
{
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<typename BT::ParamsContinuous> >(mSampleWeightsBufferId));
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<typename BT::SourceInteger> >(mClassesBufferId));
    mSampleWeights = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::ParamsContinuous> >(mSampleWeightsBufferId);
    mClasses = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::SourceInteger> >(mClassesBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mClasses->GetN())

    mStreamType = readCollection.GetBufferPtr< VectorBufferTemplate<typename BT::ParamsInteger> >(mStreamTypeBufferId);
    ASSERT_ARG_DIM_1D(mSampleWeights->GetN(), mStreamType->GetN())

    for(typename BT::Index i=0; i<mSampleWeights->GetN(); i++)
    {
        const typename BT::ParamsInteger streamType = mStreamType->Get(i);
        if(streamType == STREAM_ESTIMATION)
        {
            mStartEstimationClassHistogram.Incr(mClasses->Get(i), mSampleWeights->Get(i));
        }
        else
        {
            mStartStructureClassHistogram.Incr(mClasses->Get(i), mSampleWeights->Get(i));
        }
    }

    typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0.0);
    for(typename BT::Index c=0; c<mNumberOfClasses; c++)
    {
        const typename BT::SufficientStatsContinuous logClass = 
                mStartStructureClassHistogram.Get(c) > zero ? log2(mStartStructureClassHistogram.Get(c)) : zero;
        mStartLogStructureClassHistogram.Set(c, logClass);
    }

    mStartEntropy = calcDiscreteEntropy<BT>(mStartStructureClassHistogram.Sum(), 
                                            mStartStructureClassHistogram, 
                                            mStartLogStructureClassHistogram);   

    Reset();
}

template <class BT>
void ClassInfoGainTwoStreamWalker<BT>::Reset()
{
    mLeftEstimationClassHistogram = mStartEstimationClassHistogram;
    mLeftStructureClassHistogram = mStartStructureClassHistogram;
    mLeftLogStructureClassHistogram = mStartLogStructureClassHistogram;

    mRightEstimationClassHistogram.Zero();
    mRightStructureClassHistogram.Zero();
    mRightLogStructureClassHistogram.Zero();

    mRecomputeClassLog.clear();
}

template <class BT>
void ClassInfoGainTwoStreamWalker<BT>::MoveLeftToRight(typename BT::Index sampleIndex)
{
    const typename BT::SufficientStatsContinuous weight = mSampleWeights->Get(sampleIndex);
    const typename BT::Index classIndex = mClasses->Get(sampleIndex);
    const typename BT::ParamsInteger streamType = mStreamType->Get(sampleIndex);


    VectorBufferTemplate<typename BT::SufficientStatsContinuous>& leftClassHistogram = (streamType == STREAM_ESTIMATION) ?
                    mLeftEstimationClassHistogram : mLeftStructureClassHistogram;

    VectorBufferTemplate<typename BT::SufficientStatsContinuous>& rightClassHistogram = (streamType == STREAM_ESTIMATION) ?
                    mRightEstimationClassHistogram : mRightStructureClassHistogram;

    leftClassHistogram.Incr(classIndex, -weight);
    rightClassHistogram.Incr(classIndex, weight);
    if(streamType == STREAM_STRUCTURE)
    {
        mRecomputeClassLog[classIndex] = true;  
    }

}

template <class BT>
typename BT::ImpurityValue ClassInfoGainTwoStreamWalker<BT>::Impurity()
{
    typename BT::SufficientStatsContinuous zero = typename BT::SufficientStatsContinuous(0.0);

    for(typename BT::Index c=0; c<mNumberOfClasses; c++)
    {
        if(mRecomputeClassLog[c])
        {
            const typename BT::SufficientStatsContinuous leftLogClass = mLeftStructureClassHistogram.Get(c) > zero? log2(mLeftStructureClassHistogram.Get(c)) : zero;
            mLeftLogStructureClassHistogram.Set(c, leftLogClass);
            const typename BT::SufficientStatsContinuous rightLogClass = mRightStructureClassHistogram.Get(c) > zero ? log2(mRightStructureClassHistogram.Get(c)) : zero;
            mRightLogStructureClassHistogram.Set(c, rightLogClass);
        }
    }

    const typename BT::SufficientStatsContinuous leftWeight = mLeftStructureClassHistogram.Sum();
    const typename BT::SufficientStatsContinuous rightWeight = mRightStructureClassHistogram.Sum();
    const typename BT::SufficientStatsContinuous totalWeight = leftWeight + rightWeight;

    const typename BT::SufficientStatsContinuous leftEntropy = calcDiscreteEntropy<BT>(leftWeight, mLeftStructureClassHistogram, mLeftLogStructureClassHistogram);
    const typename BT::SufficientStatsContinuous rightEntropy = calcDiscreteEntropy<BT>(rightWeight, mRightStructureClassHistogram, mRightLogStructureClassHistogram);

    const typename BT::ImpurityValue infoGain = mStartEntropy
                                                      - ((leftWeight / totalWeight) * leftEntropy)
                                                      - ((rightWeight / totalWeight) * rightEntropy);
    return infoGain;
}

template <class BT>
typename BT::Index ClassInfoGainTwoStreamWalker<BT>::GetYDim() const
{
    return mNumberOfClasses;
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> ClassInfoGainTwoStreamWalker<BT>::GetLeftEstimationYs() const
{
    return mLeftEstimationClassHistogram.Normalized();
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> ClassInfoGainTwoStreamWalker<BT>::GetLeftStructureYs() const
{
    return mLeftStructureClassHistogram.Normalized();
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> ClassInfoGainTwoStreamWalker<BT>::GetRightEstimationYs() const
{
    return mRightEstimationClassHistogram.Normalized();
}

template <class BT>
VectorBufferTemplate<typename BT::SufficientStatsContinuous> ClassInfoGainTwoStreamWalker<BT>::GetRightStructureYs() const
{
    return mRightStructureClassHistogram.Normalized();
}


template <class BT>
typename BT::SufficientStatsContinuous ClassInfoGainTwoStreamWalker<BT>::GetLeftEstimationChildCounts() const
{
    return mLeftEstimationClassHistogram.Sum();
}

template <class BT>
typename BT::SufficientStatsContinuous ClassInfoGainTwoStreamWalker<BT>::GetLeftStructureChildCounts() const
{
    return mLeftStructureClassHistogram.Sum();
}

template <class BT>
typename BT::SufficientStatsContinuous ClassInfoGainTwoStreamWalker<BT>::GetRightEstimationChildCounts() const
{
    return mRightEstimationClassHistogram.Sum();
}

template <class BT>
typename BT::SufficientStatsContinuous ClassInfoGainTwoStreamWalker<BT>::GetRightStructureChildCounts() const
{
    return mRightStructureClassHistogram.Sum();
}