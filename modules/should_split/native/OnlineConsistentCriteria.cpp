#include <math.h>
#include "unused.h" 
#include "BufferCollectionUtils.h"
#include "OnlineConsistentCriteria.h"


OnlineConsistentCriteria::OnlineConsistentCriteria(float minImpurity,
                                                  float minNumberOfSamplesFirstSplit,
                                                  float maxNumberOfSamplesFirstSplit,
                                                  float growthRate)
: mMinImpurity(minImpurity)
, mMinNumberOfSamplesFirstSplit(minNumberOfSamplesFirstSplit)
, mMaxNumberOfSamplesFirstSplit(maxNumberOfSamplesFirstSplit)
, mGrowthRate(growthRate)
{}

OnlineConsistentCriteria::~OnlineConsistentCriteria() 
{}  

ShouldSplitCriteriaI* OnlineConsistentCriteria::Clone() const
{
    OnlineConsistentCriteria* clone = new OnlineConsistentCriteria(*this);
    return clone;
}

bool OnlineConsistentCriteria::ShouldSplit(int depth, float impurity,
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                                      BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    const float minNumberOfSamples = mMinNumberOfSamplesFirstSplit * pow(mGrowthRate, depth);
    const float maxNumberOfSamples = mMaxNumberOfSamplesFirstSplit * pow(mGrowthRate, depth);

    const bool canSplit = leftNumberOfDataponts > minNumberOfSamples 
                          && rightNumberOfDatapoints > minNumberOfSamples;
    const bool shouldSplit = canSplit && impurity > mMinImpurity;
    const bool mustSplit = canSplit && numberOfDatapoints > maxNumberOfSamples; 
    const bool result = shouldSplit || mustSplit;
    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "ShouldSplit-OnlineConsistentCriteria", nodeIndex, result ? SHOULD_SPLIT_TRUE : SHOULD_SPLIT_FALSE);
        WriteValue<int>(extraInfo, "ShouldSplit-OnlineConsistentCriteria-ShouldSplit", nodeIndex, shouldSplit ? SHOULD_SPLIT_TRUE : SHOULD_SPLIT_FALSE);
        WriteValue<int>(extraInfo, "ShouldSplit-OnlineConsistentCriteria-MustSplit", nodeIndex, mustSplit ? SHOULD_SPLIT_TRUE : SHOULD_SPLIT_FALSE);
    }
    return result;
}
