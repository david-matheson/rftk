#include <limits>

#include "unused.h" 
#include "BufferCollectionUtils.h"
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
                                      BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{

    UNUSED_PARAM(depth)
    UNUSED_PARAM(numberOfDatapoints)
    UNUSED_PARAM(leftNumberOfDataponts)
    UNUSED_PARAM(rightNumberOfDatapoints)
    bool result = impurity > (mMinImpurity +  std::numeric_limits<float>::epsilon());
    
    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "ShouldSplit-MinImpurityCriteria", nodeIndex, result ? SHOULD_SPLIT_TRUE : SHOULD_SPLIT_FALSE);
    }
    return result;
}
