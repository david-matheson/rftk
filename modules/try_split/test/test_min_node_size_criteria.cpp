#include <boost/test/unit_test.hpp>

#include "MinNodeSizeCriteria.h"

BOOST_AUTO_TEST_SUITE( MinNodeSizeCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    const int minDatapoints = 3;
    MinNodeSizeCriteria minNumberDatapoints(minDatapoints);
    BOOST_CHECK( !minNumberDatapoints.TrySplit(0, minDatapoints-1));
    BOOST_CHECK( minNumberDatapoints.TrySplit(0, minDatapoints));
    BOOST_CHECK( minNumberDatapoints.TrySplit(0, minDatapoints+1));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int minDatapoints = 3;
    TrySplitCriteriaI* minNumberDatapoints = new MinNodeSizeCriteria(minDatapoints);
    TrySplitCriteriaI* clone = minNumberDatapoints->Clone();
    delete minNumberDatapoints;

    BOOST_CHECK( !clone->TrySplit(0, minDatapoints-1));
    BOOST_CHECK( clone->TrySplit(0, minDatapoints));
    BOOST_CHECK( clone->TrySplit(0, minDatapoints+1));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()