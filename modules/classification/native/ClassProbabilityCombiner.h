#pragma once

#include "MatrixBuffer.h"

// ----------------------------------------------------------------------------
//
// Average the probabilities of each class across trees
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class ClassProbabilityCombiner
{
public:
    ClassProbabilityCombiner(int numberOfClasses);
    void Reset();
    void Combine(int nodeId, typename BufferTypes::DatapointCounts count, 
                    const MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& estimatorParameters);
    void WriteResult(int row, MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& results);
    int GetResultDim() const;

private:
    VectorBufferTemplate<typename BufferTypes::TreeEstimator> mCombinedResults;
    typename BufferTypes::DatapointCounts mNumberOfTrees;

};

template <class BufferTypes>
ClassProbabilityCombiner<BufferTypes>::ClassProbabilityCombiner(int numberOfClasses)
: mCombinedResults(numberOfClasses)
, mNumberOfTrees(typename BufferTypes::DatapointCounts(0))
{
}

template <class BufferTypes>
void ClassProbabilityCombiner<BufferTypes>::Reset()
{
    mCombinedResults.Zero();
    mNumberOfTrees = typename BufferTypes::DatapointCounts(0);
}

template <class BufferTypes>
void ClassProbabilityCombiner<BufferTypes>::Combine(int nodeId, typename BufferTypes::DatapointCounts count, 
                                                    const MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& estimatorParameters)
{
    UNUSED_PARAM(count)
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN(), estimatorParameters.GetN())
    for(int i=0; i<mCombinedResults.GetN(); i++)
    {
        mCombinedResults.Incr(i, estimatorParameters.Get(nodeId, i));
    }
    mNumberOfTrees += typename BufferTypes::DatapointCounts(1);
}

template <class BufferTypes>
void ClassProbabilityCombiner<BufferTypes>::WriteResult(int row, MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& results)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN(), results.GetN())
    typename BufferTypes::DatapointCounts numberOfTreeInv = mNumberOfTrees > typename BufferTypes::DatapointCounts(0) ? 
                            typename BufferTypes::DatapointCounts(1) / mNumberOfTrees : typename BufferTypes::DatapointCounts(0);
    for(int i=0; i<mCombinedResults.GetN(); i++)
    {
        results.Set(row, i, numberOfTreeInv * mCombinedResults.Get(i));
    }
}

template <class BufferTypes>
int ClassProbabilityCombiner<BufferTypes>::GetResultDim() const
{
    return mCombinedResults.GetN();
}
