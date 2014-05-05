#pragma once

#include <cmath>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"

#include "Constants.h"

// ----------------------------------------------------------------------------
//
// Compute the expected decrease in variance
//
// ----------------------------------------------------------------------------
template <class BT>
class SumOfVarianceImpurity
{
public:
    typename BT::SufficientStatsContinuous Impurity(int feature, int splitpoint,
                                                              const Tensor3BufferTemplate<typename BT::DatapointCounts>& childCounts,
                                                              const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& leftStats,
                                                              const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& rightStats) const;

    typedef BT BufferTypes;
};

template <class BT>
typename BT::SufficientStatsContinuous SumOfVarianceImpurity<BT>::Impurity(int feature, int splitpoint,
                                                    const Tensor3BufferTemplate<typename BT::DatapointCounts>& childCounts,
                                                    const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& leftStats,
                                                    const Tensor3BufferTemplate<typename BT::SufficientStatsContinuous>& rightStats) const
{
    const typename BT::DatapointCounts countsLeft = childCounts.Get(feature, splitpoint, LEFT_CHILD);
    const typename BT::DatapointCounts countsRight = childCounts.Get(feature, splitpoint, RIGHT_CHILD);
    const typename BT::DatapointCounts countsTotal = countsLeft+countsRight;

    typename BT::SufficientStatsContinuous startSumOfVariance = typename BT::SufficientStatsContinuous(0);
    typename BT::SufficientStatsContinuous leftSumOfVariance = typename BT::SufficientStatsContinuous(0);
    typename BT::SufficientStatsContinuous rightSumOfVariance = typename BT::SufficientStatsContinuous(0);

    const int yDim = leftStats.GetN()/2;
    for(int d=0; d<yDim; d++)
    {
        // old unstable sufficient stats 
        // const typename BT::SufficientStatsContinuous leftY = leftStats.Get(feature, splitpoint, d);
        // const typename BT::SufficientStatsContinuous leftYSquared = leftStats.Get(feature, splitpoint, d+yDim);
        // const typename BT::SufficientStatsContinuous rightY = rightStats.Get(feature, splitpoint, d);
        // const typename BT::SufficientStatsContinuous rightYSquared = rightStats.Get(feature, splitpoint, d+yDim);

        // startSumOfVariance += (leftYSquared + rightYSquared)/countsTotal -  pow((leftY + rightY) / countsTotal, 2);
        // leftSumOfVariance += (countsLeft>0.0) ? leftYSquared/countsLeft -  pow(leftY/countsLeft, 2) : 0.0;
        // rightSumOfVariance += (countsRight>0.0) ? rightYSquared/countsRight -  pow(rightY/countsRight, 2) : 0.0;

        const typename BT::SufficientStatsContinuous leftY1 = leftStats.Get(feature, splitpoint, d);
        const typename BT::SufficientStatsContinuous leftY2 = leftStats.Get(feature, splitpoint, d+yDim);
        const typename BT::SufficientStatsContinuous rightY1 = rightStats.Get(feature, splitpoint, d);
        const typename BT::SufficientStatsContinuous rightY2 = rightStats.Get(feature, splitpoint, d+yDim);

        const typename BT::SufficientStatsContinuous leftVariance = (countsLeft>0.0) ? leftY2/countsLeft : 0.0;
        const typename BT::SufficientStatsContinuous rightVariance = (countsRight>0.0) ? rightY2/countsRight : 0.0;

        const typename BT::SufficientStatsContinuous startMean = (countsLeft/countsTotal)*leftY1 + (countsRight/countsTotal)*rightY1;
        const typename BT::SufficientStatsContinuous startVariance = (countsLeft/countsTotal)*(leftVariance+leftY1*leftY1)
                                        + (countsRight/countsTotal)*(rightVariance+rightY1*rightY1)
                                        - (startMean*startMean);

        startSumOfVariance += startVariance;
        leftSumOfVariance += leftVariance;
        rightSumOfVariance += rightVariance;
    }

    const typename BT::SufficientStatsContinuous varianceGain = startSumOfVariance
                                  - ((countsLeft / countsTotal) * leftSumOfVariance)
                                  - ((countsRight / countsTotal) * rightSumOfVariance);
    return varianceGain;
}