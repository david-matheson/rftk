#pragma once

#include <time.h>

#include <MatrixBuffer.h>
#include "SplitCriteriaI.h"


class TimerSplitCriteria : public SplitCriteriaI
{
public:
    TimerSplitCriteria( SplitCriteriaI* childSplitCriteria, int minutesToRun );
    TimerSplitCriteria( SplitCriteriaI* childSplitCriteria, time_t endTime );
    TimerSplitCriteria(const TimerSplitCriteria& rhs);
    TimerSplitCriteria& operator=( const TimerSplitCriteria& rhs );
    virtual ~TimerSplitCriteria();

    virtual SplitCriteriaI* Clone() const;

    virtual bool ShouldProcessNode( int treeDepth ) const;

    virtual SPLT_CRITERIA ShouldSplit(   int treeDepth,
                                        const Float32VectorBuffer& impurityValues,
                                        const Float32MatrixBuffer& childCounts) const;

    virtual int BestSplit(  int treeDepth,
                            const Float32VectorBuffer& impurityValues,
                            const Float32MatrixBuffer& childCounts ) const;

    virtual int MinTotalSamples( int treeDepth ) const;
private:
    SplitCriteriaI* mChildSplitCriteria;
    time_t mEndTime;
};