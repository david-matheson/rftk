#include "ShouldSplitCombinedCriteria.h"

ShouldSplitCombinedCriteria::ShouldSplitCombinedCriteria(std::vector<ShouldSplitCriteriaI*> criterias)
: mCriterias()
{
    // Create a copy of each criteria
    for (std::vector<ShouldSplitCriteriaI*>::const_iterator it = criterias.begin(); it != criterias.end(); ++it)
    {
        mCriterias.push_back( (*it)->Clone() );
    }
}

ShouldSplitCombinedCriteria::~ShouldSplitCombinedCriteria()
{
    // Free each critieria
    for (std::vector<ShouldSplitCriteriaI*>::iterator it = mCriterias.begin(); it != mCriterias.end(); ++it)
    {
        delete (*it);
    }
}

ShouldSplitCriteriaI* ShouldSplitCombinedCriteria::Clone() const
{
    ShouldSplitCriteriaI* clone = new ShouldSplitCombinedCriteria(mCriterias);
    return clone;
}

bool ShouldSplitCombinedCriteria::ShouldSplit(int depth, float impurity,
                                      int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const
{
    bool trySplit = true;
    for (std::vector<ShouldSplitCriteriaI*>::const_iterator it = mCriterias.begin(); it != mCriterias.end(); ++it)
    {
        trySplit = trySplit && (*it)->ShouldSplit(depth, impurity, numberOfDatapoints, leftNumberOfDataponts, rightNumberOfDatapoints);
    }
    return trySplit;

}
