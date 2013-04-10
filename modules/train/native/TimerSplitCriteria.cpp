#include <time.h>

#include <asserts.h>
#include "TimerSplitCriteria.h"

TimerSplitCriteria::TimerSplitCriteria( SplitCriteriaI* childSplitCriteria, int secondsToRun )
: mChildSplitCriteria(childSplitCriteria->Clone())
, mEndTime( time(NULL) + secondsToRun)
{
}

TimerSplitCriteria::TimerSplitCriteria( SplitCriteriaI* childSplitCriteria, time_t endTime )
: mChildSplitCriteria(childSplitCriteria->Clone())
, mEndTime(endTime)
{
}

TimerSplitCriteria::TimerSplitCriteria(const TimerSplitCriteria& rhs)
: mChildSplitCriteria(rhs.mChildSplitCriteria->Clone())
, mEndTime(rhs.mEndTime)
{
}

TimerSplitCriteria& TimerSplitCriteria::operator=( const TimerSplitCriteria& rhs )
{
    delete mChildSplitCriteria;
    mChildSplitCriteria = rhs.mChildSplitCriteria->Clone();
    mEndTime = rhs.mEndTime;
    return *this;
}


TimerSplitCriteria::~TimerSplitCriteria()
{
    delete mChildSplitCriteria;
}

SplitCriteriaI* TimerSplitCriteria::Clone() const
{
    return new TimerSplitCriteria(mChildSplitCriteria->Clone(), mEndTime);
}

bool TimerSplitCriteria::ShouldProcessNode( int treeDepth ) const
{
    const time_t now = time( NULL);
    return ((now < mEndTime) && mChildSplitCriteria->ShouldProcessNode(treeDepth));
}

SPLT_CRITERIA TimerSplitCriteria::ShouldSplit(   int treeDepth,
                                    const Float32VectorBuffer& impurityValues,
                                    const Float32MatrixBuffer& childCounts) const
{
    return mChildSplitCriteria->ShouldSplit(treeDepth, impurityValues, childCounts);
}

int TimerSplitCriteria::BestSplit(  int treeDepth,
                        const Float32VectorBuffer& impurityValues,
                        const Float32MatrixBuffer& childCounts ) const
{
    return mChildSplitCriteria->BestSplit(treeDepth, impurityValues, childCounts);
}

int TimerSplitCriteria::MinTotalSamples( int treeDepth ) const
{
    return mChildSplitCriteria->MinTotalSamples(treeDepth);
}
