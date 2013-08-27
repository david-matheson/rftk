#include "unused.h" 
#include "BufferCollectionUtils.h"
#include "TrySplitNoCriteria.h"


TrySplitNoCriteria::TrySplitNoCriteria()
{}

TrySplitNoCriteria::~TrySplitNoCriteria()
{}

TrySplitCriteriaI* TrySplitNoCriteria::Clone() const
{
    TrySplitNoCriteria* clone = new TrySplitNoCriteria(*this);
    return clone;
}

bool TrySplitNoCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(depth);
    UNUSED_PARAM(numberOfDatapoints);
    UNUSED_PARAM(extraInfo)
    UNUSED_PARAM(nodeIndex)
    const bool trySplit = true;
    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "TrySplit-TrySplitNoCriteria", nodeIndex, trySplit ? TRY_SPLIT_TRUE : TRY_SPLIT_FALSE);      
    }
    return trySplit;
}
