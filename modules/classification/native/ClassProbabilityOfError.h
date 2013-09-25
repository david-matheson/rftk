#pragma once

#include <Tree.h>

// ----------------------------------------------------------------------------
//
// Probability of error for a node in the tree
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class ClassProbabilityOfError
{
public:
    typename BufferTypes::TreeEstimator ProbabilityOfError(const Tree& tree, const int nodeIndex) const;
};

template <class BufferTypes>
typename BufferTypes::TreeEstimator ClassProbabilityOfError<BufferTypes>::ProbabilityOfError(const Tree& tree, const int nodeIndex) const
{
    const typename BufferTypes::TreeEstimator* ys = tree.GetYs().GetRowPtrUnsafe(nodeIndex);
    const typename BufferTypes::TreeEstimator probabilityOfError = 1.0f - *std::max_element(ys, ys + tree.GetYs().GetN());
    return probabilityOfError;
}