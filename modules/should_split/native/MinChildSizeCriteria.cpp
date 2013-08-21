#include "asserts.h" // for UNUSED_PARAM
#include "MinChildSizeCriteria.h"


MinChildSizeCriteria::MinChildSizeCriteria(int minNumberOfChildDatapoints)
: mMinNumberOfChildDatapoints(minNumberOfChildDatapoints)
{}

MinChildSizeCriteria::~MinChildSizeCriteria() 
{}  

ShouldSplitCriteriaI* MinChildSizeCriteria::Clone() const
{
    MinChildSizeCriteria* clone = new MinChildSizeCriteria(*this);
    return clone;
}

bool MinChildSizeCriteria::ShouldSplit(int depth, float impurity,
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                                      BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(extraInfo)
    UNUSED_PARAM(nodeIndex)
    return (leftNumberOfDataponts >= mMinNumberOfChildDatapoints) 
          && (rightNumberOfDatapoints >= mMinNumberOfChildDatapoints);
}
