#pragma once

#include "SplitCriteriaI.h"

class AlphaBetaSplitCriteria : public SplitCriteriaI
{
    virtual bool ShouldSplit();
};