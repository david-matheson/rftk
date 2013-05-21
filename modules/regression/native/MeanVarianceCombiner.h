#pragma once

#include "MatrixBuffer.h"

// ----------------------------------------------------------------------------
//
// Combine the mean and variance across trees
//
// ----------------------------------------------------------------------------
template <class FloatType>
class MeanVarianceCombiner
{
public:
    MeanVarianceCombiner(int numberOfDimensions);
    void Reset();
    void Combine(int nodeId, FloatType count, const MatrixBufferTemplate<FloatType>& estimatorParameters);
    void WriteResult(int row, MatrixBufferTemplate<FloatType>& results);
    int GetResultDim() const;

private:
    VectorBufferTemplate<FloatType> mCombinedResults;
    FloatType mNumberOfTrees;
    FloatType mCounts;
};

template <class FloatType>
MeanVarianceCombiner<FloatType>::MeanVarianceCombiner(int numberOfDimensions)
: mCombinedResults(numberOfDimensions*2)
, mNumberOfTrees(FloatType(0))
, mCounts(FloatType(0))
{
}

template <class FloatType>
void MeanVarianceCombiner<FloatType>::Reset()
{
    mCombinedResults.Zero();
    mNumberOfTrees = FloatType(0);
    mCounts = FloatType(0);
}

template <class FloatType>
void MeanVarianceCombiner<FloatType>::Combine(int nodeId, FloatType count, const MatrixBufferTemplate<FloatType>& estimatorParameters)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN(), estimatorParameters.GetN())
    for(int i=0; i<mCombinedResults.GetN(); i++)
    {
        mCombinedResults.Incr(i, estimatorParameters.Get(nodeId, i));

    }
    mNumberOfTrees += FloatType(1);
    mCounts += count;
}

template <class FloatType>
void MeanVarianceCombiner<FloatType>::WriteResult(int row, MatrixBufferTemplate<FloatType>& results)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN()/2, results.GetN())
    FloatType numberOfTreesInv = mNumberOfTrees > FloatType(0) ? FloatType(1) / mNumberOfTrees : FloatType(0);
    for(int i=0; i<results.GetN(); i++)
    {
        results.Set(row, i, numberOfTreesInv * mCombinedResults.Get(i));
    }
}

template <class FloatType>
int MeanVarianceCombiner<FloatType>::GetResultDim() const
{
    return mCombinedResults.GetN()/2;
}
