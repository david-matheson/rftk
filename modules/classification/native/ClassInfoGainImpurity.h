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
template <class BT>
class ClassInfoGainImpurity
{
public:
    typename BT::ImpurityValue Impurity(int feature, int splitpoint,
                      const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& childCounts,
                      const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& leftStats,
                      const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& rightStats) const;


    typedef BT BufferTypes;
};

template <class BT>
typename BT::ImpurityValue ClassInfoGainImpurity<BT>::Impurity(int feature, int splitpoint,
                                                    const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& childCounts,
                                                    const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& leftStats,
                                                    const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& rightStats) const
{
    const typename BT::SufficientStatsContinuous zero(0);
    const typename BT::SufficientStatsContinuous one(1);

    const typename BT::SufficientStatsContinuous countsLeft = childCounts.Get(feature, splitpoint, LEFT_CHILD);
    const typename BT::SufficientStatsContinuous countsRight = childCounts.Get(feature, splitpoint, RIGHT_CHILD);

    const typename BT::SufficientStatsContinuous invCountsLeft = one / countsLeft;
    const typename BT::SufficientStatsContinuous invCountsRight = one / countsRight;
    const typename BT::SufficientStatsContinuous countsTotal = countsLeft+countsRight;
    const typename BT::SufficientStatsContinuous invCountsTotal = one / countsTotal;

    typename BT::SufficientStatsContinuous startEntropy = zero;
    typename BT::SufficientStatsContinuous leftEntropy = zero;
    typename BT::SufficientStatsContinuous rightEntropy = zero;

    for(typename BT::Index classId=0; classId<leftStats.GetN(); classId++)
    {
        const typename BT::SufficientStatsContinuous left = leftStats.Get(feature, splitpoint, classId);
        const typename BT::SufficientStatsContinuous right = rightStats.Get(feature, splitpoint, classId);
        const typename BT::SufficientStatsContinuous combined = left + right;
        startEntropy -= (combined > std::numeric_limits<typename BT::SufficientStatsContinuous>::epsilon()) ?
                              (combined*invCountsTotal) * log2(combined*invCountsTotal) : typename BT::SufficientStatsContinuous(0);
        leftEntropy -= (left > std::numeric_limits<typename BT::SufficientStatsContinuous>::epsilon()) ?
                              (left*invCountsLeft) * log2(left*invCountsLeft) : typename BT::SufficientStatsContinuous(0);
        rightEntropy -= (right > std::numeric_limits<typename BT::SufficientStatsContinuous>::epsilon()) ?
                              (right*invCountsRight) * log2(right*invCountsRight) : typename BT::SufficientStatsContinuous(0);
    }

    const typename BT::ImpurityValue infoGain = startEntropy
                                                      - ((countsLeft / countsTotal) * leftEntropy)
                                                      - ((countsRight / countsTotal) * rightEntropy);
    return infoGain;
}