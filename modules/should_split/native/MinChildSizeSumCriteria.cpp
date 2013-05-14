#include "asserts.h" // for UNUSED_PARAM
#include "MinChildSizeSumCriteria.h"


MinChildSizeSumCriteria::MinChildSizeSumCriteria(int minNumberOfChildDatapointsSum)
: mMinNumberOfChildDatapointsSum(minNumberOfChildDatapointsSum)
{}

MinChildSizeSumCriteria::~MinChildSizeSumCriteria() 
{}  

ShouldSplitCriteriaI* MinChildSizeSumCriteria::Clone() const
{
    MinChildSizeSumCriteria* clone = new MinChildSizeSumCriteria(*this);
    return clone;
}

bool MinChildSizeSumCriteria::ShouldSplit(int depth, float impurity,
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    return (leftNumberOfDataponts + rightNumberOfDatapoints)
                   >= mMinNumberOfChildDatapointsSum;
}
