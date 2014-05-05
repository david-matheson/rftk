#include "unused.h" 
#include "BufferCollectionUtils.h"
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
                                      BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    bool result = (leftNumberOfDataponts >= mMinNumberOfChildDatapoints) 
          && (rightNumberOfDatapoints >= mMinNumberOfChildDatapoints);

    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "ShouldSplit-MinChildSizeCriteria", nodeIndex, result ? SHOULD_SPLIT_TRUE : SHOULD_SPLIT_FALSE);      
    }

    return result;
}
