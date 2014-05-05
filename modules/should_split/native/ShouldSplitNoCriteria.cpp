#include "unused.h" 
#include "BufferCollectionUtils.h"
#include "ShouldSplitNoCriteria.h"


ShouldSplitNoCriteria::ShouldSplitNoCriteria()
{}

ShouldSplitNoCriteria::~ShouldSplitNoCriteria() 
{}  

ShouldSplitCriteriaI* ShouldSplitNoCriteria::Clone() const
{
    ShouldSplitNoCriteria* clone = new ShouldSplitNoCriteria(*this);
    return clone;
}

bool ShouldSplitNoCriteria::ShouldSplit(int depth, float impurity,
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                                      BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(leftNumberOfDataponts)
    UNUSED_PARAM(rightNumberOfDatapoints)
    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "ShouldSplit-ShouldSplitNoCriteria", nodeIndex, SHOULD_SPLIT_TRUE);
    }
    return true;
}
