#include "BufferCollectionUtils.h"
#include "TrySplitCombinedCriteria.h"

TrySplitCombinedCriteria::TrySplitCombinedCriteria(std::vector<TrySplitCriteriaI*> criterias)
: mCriterias()
{
    // Create a copy of each criteria
    for (std::vector<TrySplitCriteriaI*>::const_iterator it = criterias.begin(); it != criterias.end(); ++it)
    {
        mCriterias.push_back( (*it)->Clone() );
    }
}

TrySplitCombinedCriteria::~TrySplitCombinedCriteria()
{
    // Free each critieria
    for (std::vector<TrySplitCriteriaI*>::iterator it = mCriterias.begin(); it != mCriterias.end(); ++it)
    {
        delete (*it);
    }
}

TrySplitCriteriaI* TrySplitCombinedCriteria::Clone() const
{
    TrySplitCriteriaI* clone = new TrySplitCombinedCriteria(mCriterias);
    return clone;
}

bool TrySplitCombinedCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    bool trySplit = true;
    for (std::vector<TrySplitCriteriaI*>::const_iterator it = mCriterias.begin(); it != mCriterias.end(); ++it)
    {
        trySplit = trySplit && (*it)->TrySplit(depth, numberOfDatapoints, extraInfo, nodeIndex, recordInfo);
    }
    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "TrySplit-TrySplitCombinedCriteria", nodeIndex, trySplit ? TRY_SPLIT_TRUE : TRY_SPLIT_FALSE);      
    }
    return trySplit;
}
