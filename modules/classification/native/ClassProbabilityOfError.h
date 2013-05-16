#pragma once

#include <Tree.h>

// ----------------------------------------------------------------------------
//
// Probability of error for a node in the tree
//
// ----------------------------------------------------------------------------
class ClassProbabilityOfError
{
public:
    float ProbabilityOfError(const Tree& tree, const int nodeIndex) const;
};

float ClassProbabilityOfError::ProbabilityOfError(const Tree& tree, const int nodeIndex) const
{
    const float* ys = tree.mYs.GetRowPtrUnsafe(nodeIndex);
    const float probabilityOfError = 1.0f - *std::max_element(ys, ys + tree.mYs.GetN());
    return probabilityOfError;
}