#include <math.h>
#include <asserts.h> // for UNUSED_PARAM
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
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const
{

    const float minNumberOfSamples = mMinNumberOfSamplesFirstSplit * pow(mGrowthRate, depth);
    const float maxNumberOfSamples = mMaxNumberOfSamplesFirstSplit * pow(mGrowthRate, depth);

    const bool canSplit = leftNumberOfDataponts > minNumberOfSamples 
                          && rightNumberOfDatapoints > minNumberOfSamples;
    const bool shouldSplit = canSplit && impurity > mMinImpurity;
    const bool mustSplit = canSplit && numberOfDatapoints > maxNumberOfSamples; 

    return shouldSplit || mustSplit;
}
