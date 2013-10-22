#pragma once

#include "MatrixBuffer.h"
#include "Tree.h"

// ----------------------------------------------------------------------------
//
// Expected error of variance
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class SumOfVarianceExpectedError
{
public:
    typename BufferTypes::TreeEstimator ProbabilityOfError(const Tree& tree, const int nodeIndex) const;
};

template <class BufferTypes>
typename BufferTypes::TreeEstimator SumOfVarianceExpectedError<BufferTypes>::ProbabilityOfError(const Tree& tree, const int nodeIndex) const
{
    const MatrixBufferTemplate< typename BufferTypes::TreeEstimator >& ys = tree.GetYs();
    const typename BufferTypes::TreeEstimator count = tree.GetCounts().Get(nodeIndex);
    const int yDim = ys.GetN() / 2;
    typename BufferTypes::TreeEstimator sumOfVariance = 0;

    for(int d=yDim; d<ys.GetN(); d++)
    {
        sumOfVariance += (count>0.0) ? ys.Get(nodeIndex, d)/count : typename BufferTypes::TreeEstimator(0);
    }

    return sumOfVariance;
}