#pragma once

#include <time.h>

#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// TimeLimitCriteria checks if a time limit has been exceeded
//
// ----------------------------------------------------------------------------
class TimeLimitCriteria: public TrySplitCriteriaI
{
public:
    TimeLimitCriteria(int secondsToRuns);
    TimeLimitCriteria(time_t endTime);
    virtual ~TimeLimitCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, int numberOfDatapoints) const;
private:
    const time_t mEndTime;
};