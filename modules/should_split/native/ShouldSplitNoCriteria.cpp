#include "asserts.h" // for UNUSED_PARAM
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
                                      BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(leftNumberOfDataponts)
    UNUSED_PARAM(rightNumberOfDatapoints)
    UNUSED_PARAM(extraInfo)
    UNUSED_PARAM(nodeIndex)
    return true;
}
