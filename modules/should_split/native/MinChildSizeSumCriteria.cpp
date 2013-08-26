#include "unused.h"
#include "BufferCollectionUtils.h"
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
                                      BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    bool result = (leftNumberOfDataponts + rightNumberOfDatapoints)
                   >= mMinNumberOfChildDatapointsSum;

    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "ShouldSplit-MinChildSizeSumCriteria", nodeIndex, result ? SHOULD_SPLIT_TRUE : SHOULD_SPLIT_FALSE);
    }
    return result;
}
