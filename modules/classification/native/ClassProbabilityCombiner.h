#pragma once

#include "MatrixBuffer.h"

// ----------------------------------------------------------------------------
//
// Average the probabilities of each class across trees
//
// ----------------------------------------------------------------------------
template <class FloatType>
class ClassProbabilityCombiner
{
public:
    ClassProbabilityCombiner(int numberOfClasses);
    void Reset();
    void Combine(int nodeId, const MatrixBufferTemplate<FloatType>& estimatorParameters);
    void WriteResult(int row, MatrixBufferTemplate<FloatType>& results);

private:
    VectorBufferTemplate<FloatType> mCombinedResults;
    FloatType mNumberOfTrees;

};

template <class FloatType>
ClassProbabilityCombiner<FloatType>::ClassProbabilityCombiner(int numberOfClasses)
: mCombinedResults(numberOfClasses)
, mNumberOfTrees(FloatType(0))
{
}

template <class FloatType>
void ClassProbabilityCombiner<FloatType>::Reset()
{
    mCombinedResults.Zero();
    mNumberOfTrees = FloatType(0);
}

template <class FloatType>
void ClassProbabilityCombiner<FloatType>::Combine(int nodeId, const MatrixBufferTemplate<FloatType>& estimatorParameters)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN(), estimatorParameters.GetN())
    for(int i=0; i<mCombinedResults.GetN(); i++)
    {
        mCombinedResults.Incr(i, estimatorParameters.Get(nodeId, i));
    }
    mNumberOfTrees += FloatType(1);
}

template <class FloatType>
void ClassProbabilityCombiner<FloatType>::WriteResult(int row, MatrixBufferTemplate<FloatType>& results)
{
    ASSERT_ARG_DIM_1D(mCombinedResults.GetN(), results.GetN())
    FloatType numberOfTreeInv = mNumberOfTrees > FloatType(0) ? FloatType(1) / mNumberOfTrees : FloatType(0);
    for(int i=0; i<mCombinedResults.GetN(); i++)
    {
        results.Set(row, i, numberOfTreeInv * mCombinedResults.Get(i));
    }
}
