#include "asserts.h" // for UNUSED_PARAM
#include "MinNumberDatapointsCriteria.h"


MinNumberDatapointsCriteria::MinNumberDatapointsCriteria(int minNumberOfDatapoints)
: mMinNumberOfDatapoints(minNumberOfDatapoints)
{}

MinNumberDatapointsCriteria::~MinNumberDatapointsCriteria()
{}

TrySplitCriteriaI* MinNumberDatapointsCriteria::Clone() const
{
    MinNumberDatapointsCriteria* clone = new MinNumberDatapointsCriteria(*this);
    return clone;
}

bool MinNumberDatapointsCriteria::TrySplit(int depth, int numberOfDatapoints) const
{
    UNUSED_PARAM(depth);
    return (numberOfDatapoints >= mMinNumberOfDatapoints);
}
