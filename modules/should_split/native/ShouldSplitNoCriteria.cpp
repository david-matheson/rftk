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
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const
{
    UNUSED_PARAM(depth)
    UNUSED_PARAM(impurity)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(leftNumberOfDataponts)
    UNUSED_PARAM(rightNumberOfDatapoints)
    return true;
}
