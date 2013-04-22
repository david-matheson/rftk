#include "CombinedCriteria.h"

CombinedCriteria::CombinedCriteria(std::vector<TrySplitCriteriaI*> criterias)
: mCriterias()
{
    // Create a copy of each criteria
    for (std::vector<TrySplitCriteriaI*>::const_iterator it = criterias.begin(); it != criterias.end(); ++it)
    {
        mCriterias.push_back( (*it)->Clone() );
    }
}

CombinedCriteria::~CombinedCriteria()
{
    // Free each critieria
    for (std::vector<TrySplitCriteriaI*>::iterator it = mCriterias.begin(); it != mCriterias.end(); ++it)
    {
        delete (*it);
    }
}

TrySplitCriteriaI* CombinedCriteria::Clone() const
{
    TrySplitCriteriaI* clone = new CombinedCriteria(mCriterias);
    return clone;
}

bool CombinedCriteria::TrySplit(int depth, int numberOfDatapoints) const
{
    bool trySplit = true;
    for (std::vector<TrySplitCriteriaI*>::const_iterator it = mCriterias.begin(); it != mCriterias.end(); ++it)
    {
        trySplit = trySplit && (*it)->TrySplit(depth, numberOfDatapoints);
    }
    return trySplit;

}
