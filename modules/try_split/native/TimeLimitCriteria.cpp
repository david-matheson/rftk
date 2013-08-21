#include "asserts.h" // for UNUSED_PARAM
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

bool TimeLimitCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(depth);
    UNUSED_PARAM(numberOfDatapoints);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    const time_t now = time(NULL);
    return (now < mEndTime);
}
