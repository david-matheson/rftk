#include "unused.h" 
#include "BufferCollectionUtils.h"
#include "TimeLimitCriteria.h"

TimeLimitCriteria::TimeLimitCriteria(int secondsToRuns)
: mEndTime( time(NULL) + secondsToRuns )
{}

TimeLimitCriteria::TimeLimitCriteria(time_t endTime)
: mEndTime( endTime )
{}

TimeLimitCriteria::~TimeLimitCriteria()
{}

TrySplitCriteriaI* TimeLimitCriteria::Clone() const
{
    TrySplitCriteriaI* clone = new TimeLimitCriteria(mEndTime);
    return clone;
}

bool TimeLimitCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(depth);
    UNUSED_PARAM(numberOfDatapoints);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    const time_t now = time(NULL);
    const bool result = (now < mEndTime);

    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "TrySplit-TimeLimitCriteria", nodeIndex, result ? TRY_SPLIT_TRUE : TRY_SPLIT_FALSE);      
    }

    return result;
}
