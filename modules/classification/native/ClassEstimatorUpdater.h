#pragma once

#include <VectorBuffer.h>
#include <MatrixBuffer.h>
#include <BufferCollectionStack.h>
#include <UniqueBufferId.h>
#include <Tree.h>

// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class BindedClassEstimatorUpdater
{
public:
    BindedClassEstimatorUpdater();

    BindedClassEstimatorUpdater(VectorBufferTemplate<FloatType> const* sampleWeights,
                            VectorBufferTemplate<IntType> const* mClasses,
                            int numberOfClasses);

    void UpdateEstimator(Tree& tree, 
                              const int nodeIndex,
                              int sampleIndex ) const;


private:
    VectorBufferTemplate<FloatType> const* mSampleWeights;
    VectorBufferTemplate<IntType> const* mClasses;
    int mNumberOfClasses;
};

template <class FloatType, class IntType>
BindedClassEstimatorUpdater<FloatType, IntType>::BindedClassEstimatorUpdater()
: mSampleWeights(NULL)
, mClasses(NULL)
, mNumberOfClasses(0)
{}


template <class FloatType, class IntType>
BindedClassEstimatorUpdater<FloatType, IntType>::BindedClassEstimatorUpdater(
                                                              VectorBufferTemplate<FloatType> const* sampleWeights,
                                                              VectorBufferTemplate<IntType> const* classes,
                                                              int numberOfClasses)
: mSampleWeights(sampleWeights)
, mClasses(classes)
, mNumberOfClasses(numberOfClasses)
{}

template <class FloatType, class IntType>
void BindedClassEstimatorUpdater<FloatType, IntType>::UpdateEstimator(Tree& tree, 
                                                                                  const int nodeIndex,
                                                                                  int sampleIndex ) const
{
    const FloatType weight = mSampleWeights->Get(sampleIndex);
    const IntType classId = mClasses->Get(sampleIndex);

    const float oldN = tree.mCounts.Get(nodeIndex);
    for(int c=0; c<mNumberOfClasses; c++)
    {
        float classCount = oldN * tree.mYs.Get(nodeIndex,c);
        if( classId == c )
        {
            classCount += weight;
        }
        const float updatedClassProbability = classCount / (oldN + weight);
        tree.mYs.Set(nodeIndex, c, updatedClassProbability);
    }
}


// ----------------------------------------------------------------------------
//
// Update class histograms from sample weights and classes
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class ClassEstimatorUpdater
{
public:
    ClassEstimatorUpdater(const BufferId& sampleWeightsBufferId,
                       const BufferId& classesBufferId,
                       const int numberOfClasses);

    BindedClassEstimatorUpdater<FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const;
    int GetNumberOfClasses() const;

    typedef FloatType Float;
    typedef IntType Int;
    typedef BindedClassEstimatorUpdater<FloatType, IntType> BindedEstimatorUpdater;

private:
    const BufferId mSampleWeightsBufferId;
    const BufferId mClassesBufferId;
    const int mNumberOfClasses;
};

template <class FloatType, class IntType>
ClassEstimatorUpdater<FloatType, IntType>::ClassEstimatorUpdater(const BufferId& sampleWeightsBufferId,
                                                          const BufferId& classesBufferId,
                                                          const int numberOfClasses)
: mSampleWeightsBufferId(sampleWeightsBufferId)
, mClassesBufferId(classesBufferId)
, mNumberOfClasses(numberOfClasses)
{}

template <class FloatType, class IntType>
BindedClassEstimatorUpdater<FloatType, IntType> 
ClassEstimatorUpdater<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection) const
{
    VectorBufferTemplate<FloatType> const* sampleWeights = 
          readCollection.GetBufferPtr< VectorBufferTemplate<FloatType> >(mSampleWeightsBufferId);

    VectorBufferTemplate<IntType> const* classes = 
          readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mClassesBufferId);

    return BindedClassEstimatorUpdater<FloatType, IntType>(sampleWeights, classes, mNumberOfClasses);
}

template <class FloatType, class IntType>
int ClassEstimatorUpdater<FloatType, IntType>::GetNumberOfClasses() const
{
    return mNumberOfClasses;
}
