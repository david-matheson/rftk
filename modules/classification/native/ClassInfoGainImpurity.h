#pragma once

#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"

#include "Constants.h"
#include "ClassEntropyUtils.h"

// ----------------------------------------------------------------------------
//
// Compute the class information gain for split statistics
//
// ----------------------------------------------------------------------------
template <class FloatType>
class ClassInfoGainImpurity
{
public:
    FloatType Impurity(int feature, int splitpoint,
                      const Tensor3BufferTemplate<FloatType>& childCounts,
                      const Tensor3BufferTemplate<FloatType>& leftStats,
                      const Tensor3BufferTemplate<FloatType>& rightStats) const;

    typedef FloatType Float;
};

template <class FloatType>
FloatType ClassInfoGainImpurity<FloatType>::Impurity(int feature, int splitpoint,
                                                    const Tensor3BufferTemplate<FloatType>& childCounts,
                                                    const Tensor3BufferTemplate<FloatType>& leftStats,
                                                    const Tensor3BufferTemplate<FloatType>& rightStats) const
{
    const FloatType countsLeft = childCounts.Get(feature, splitpoint, LEFT_CHILD);
    const FloatType countsRight = childCounts.Get(feature, splitpoint, RIGHT_CHILD);

    const FloatType invCountsLeft = FloatType(1) / countsLeft;
    const FloatType invCountsRight = FloatType(1) / countsRight;
    const FloatType countsTotal = countsLeft+countsRight;
    const FloatType invCountsTotal = FloatType(1) / countsTotal;

    FloatType startEntropy = FloatType(0);
    FloatType leftEntropy = FloatType(0);
    FloatType rightEntropy = FloatType(0);

    for(int classId=0; classId<leftStats.GetN(); classId++)
    {
        const FloatType left = leftStats.Get(feature, splitpoint, classId);
        const FloatType right = rightStats.Get(feature, splitpoint, classId);
        const FloatType combined = left + right;
        startEntropy -= (combined > std::numeric_limits<FloatType>::epsilon()) ?
                              (combined*invCountsTotal) * log2(combined*invCountsTotal) : FloatType(0);
        leftEntropy -= (left > std::numeric_limits<FloatType>::epsilon()) ?
                              (left*invCountsLeft) * log2(left*invCountsLeft) : FloatType(0);
        rightEntropy -= (right > std::numeric_limits<FloatType>::epsilon()) ?
                              (right*invCountsRight) * log2(right*invCountsRight) : FloatType(0);
    }

    const FloatType infoGain = startEntropy
                                  - ((countsLeft / countsTotal) * leftEntropy)
                                  - ((countsRight / countsTotal) * rightEntropy);
    return infoGain;
}