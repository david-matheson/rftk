#include <limits>

#include "asserts.h" // for UNUSED_PARAM
#include "MinImpurityCriteria.h"


MinImpurityCriteria::MinImpurityCriteria(float minImpurity)
: mMinImpurity(minImpurity)
{}

MinImpurityCriteria::~MinImpurityCriteria() 
{}  

ShouldSplitCriteriaI* MinImpurityCriteria::Clone() const
{
    MinImpurityCriteria* clone = new MinImpurityCriteria(*this);
    return clone;
}

bool MinImpurityCriteria::ShouldSplit(int depth, float impurity,
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                                      BufferCollection& extraInfo, int nodeIndex) const
{

    UNUSED_PARAM(depth)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(leftNumberOfDataponts)
    UNUSED_PARAM(rightNumberOfDatapoints)
    UNUSED_PARAM(extraInfo)
    UNUSED_PARAM(nodeIndex)
    return impurity > (mMinImpurity +  std::numeric_limits<float>::epsilon());
}
