#include <boost/test/unit_test.hpp>

#include "MinNumberDatapointsCriteria.h"

BOOST_AUTO_TEST_SUITE( MinNumberDatapointsCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    const int minDatapoints = 3;
    MinNumberDatapointsCriteria minNumberDatapoints(minDatapoints);
    BOOST_CHECK( !minNumberDatapoints.TrySplit(0, minDatapoints-1));
    BOOST_CHECK( minNumberDatapoints.TrySplit(0, minDatapoints));
    BOOST_CHECK( minNumberDatapoints.TrySplit(0, minDatapoints+1));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int minDatapoints = 3;
    TrySplitCriteriaI* minNumberDatapoints = new MinNumberDatapointsCriteria(minDatapoints);
    TrySplitCriteriaI* clone = minNumberDatapoints->Clone();
    delete minNumberDatapoints;

    BOOST_CHECK( !clone->TrySplit(0, minDatapoints-1));
    BOOST_CHECK( clone->TrySplit(0, minDatapoints));
    BOOST_CHECK( clone->TrySplit(0, minDatapoints+1));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()