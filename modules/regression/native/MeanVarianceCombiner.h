#pragma once

#include "MatrixBuffer.h"

// ----------------------------------------------------------------------------
//
// Combine the mean and variance across trees
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class MeanVarianceCombiner
{
public:
    MeanVarianceCombiner(int numberOfDimensions);
    void Reset();
    void Combine(int nodeId, typename BufferTypes::DatapointCounts count, 
                    const MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& estimatorParameters,
                    double weight);
    void WriteResult(int row, MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& results);
    int GetResultDim() const;

private:
    VectorBufferTemplate<typename BufferTypes::TreeEstimator> mCombinedResults;
    typename BufferTypes::DatapointCounts mNumberOfTrees;
    typename BufferTypes::DatapointCounts mCounts;
};

template <class BufferTypes>
MeanVarianceCombiner<BufferTypes>::MeanVarianceCombiner(int numberOfDimensions)
: mCombinedResults(numberOfDimensions*2)
, mNumberOfTrees(typename BufferTypes::DatapointCounts(0))
, mCounts(typename BufferTypes::DatapointCounts(0))
{
}

template <class BufferTypes>
void MeanVarianceCombiner<BufferTypes>::Reset()
{
    mCombinedResults.Zero();
    mNumberOfTrees = typename BufferTypes::DatapointCounts(0);
    mCounts = typename BufferTypes::DatapointCounts(0);
}

template <class BufferTypes>
void MeanVarianceCombiner<BufferTypes>::Combine(int nodeId, typename BufferTypes::DatapointCounts count, 
                                                    const MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& estimatorParameters,
                                                    double weight)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN(), estimatorParameters.GetN())
    for(int i=0; i<mCombinedResults.GetN(); i++)
    {
        mCombinedResults.Incr(i, weight*estimatorParameters.Get(nodeId, i));

    }
    mNumberOfTrees += (count > 0.1) ? typename BufferTypes::DatapointCounts(weight) : typename BufferTypes::DatapointCounts(0);
    mCounts += count*weight;
}

template <class BufferTypes>
void MeanVarianceCombiner<BufferTypes>::WriteResult(int row, MatrixBufferTemplate<typename BufferTypes::TreeEstimator>& results)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN()/2, results.GetN())
    typename BufferTypes::DatapointCounts numberOfTreesInv = mNumberOfTrees > typename BufferTypes::DatapointCounts(0) ? 
                                typename BufferTypes::DatapointCounts(1) / mNumberOfTrees : typename BufferTypes::DatapointCounts(0);
    for(int i=0; i<results.GetN(); i++)
    {
        results.Set(row, i, numberOfTreesInv * mCombinedResults.Get(i));
    }
}

template <class BufferTypes>
int MeanVarianceCombiner<BufferTypes>::GetResultDim() const
{
    return mCombinedResults.GetN()/2;
}
