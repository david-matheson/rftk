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
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                                      BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(extraInfo)
    UNUSED_PARAM(nodeIndex)
    return (leftNumberOfDataponts + rightNumberOfDatapoints)
                   >= mMinNumberOfChildDatapointsSum;
}
