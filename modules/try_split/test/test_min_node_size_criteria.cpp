#include <boost/test/unit_test.hpp>

#include "MinNodeSizeCriteria.h"

BOOST_AUTO_TEST_SUITE( MinNodeSizeCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    const float minDatapoints = 3;
    MinNodeSizeCriteria minNumberDatapoints(minDatapoints);
    BufferCollection bc;
    BOOST_CHECK( !minNumberDatapoints.TrySplit(0, minDatapoints-1, bc, 0));
    BOOST_CHECK( minNumberDatapoints.TrySplit(0, minDatapoints, bc, 0));
    BOOST_CHECK( minNumberDatapoints.TrySplit(0, minDatapoints+1, bc, 0));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const float minDatapoints = 3;
    TrySplitCriteriaI* minNumberDatapoints = new MinNodeSizeCriteria(minDatapoints);
    TrySplitCriteriaI* clone = minNumberDatapoints->Clone();
    delete minNumberDatapoints;
    BufferCollection bc;
    BOOST_CHECK( !clone->TrySplit(0, minDatapoints-1, bc, 0));
    BOOST_CHECK( clone->TrySplit(0, minDatapoints, bc, 0));
    BOOST_CHECK( clone->TrySplit(0, minDatapoints+1, bc, 0));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()