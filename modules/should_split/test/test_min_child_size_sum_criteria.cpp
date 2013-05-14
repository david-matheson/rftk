#include <boost/test/unit_test.hpp>

#include "MinChildSizeSumCriteria.h"

BOOST_AUTO_TEST_SUITE( MinChildSizeSumCriteriaTests )

BOOST_AUTO_TEST_CASE(test_ShouldSplit)
{
    const int minChildSize = 6;
    MinChildSizeSumCriteria minChildSizeCriteria(minChildSize);
    BOOST_CHECK( minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize/2, minChildSize/2));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize/2-1, minChildSize/2));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize/2, minChildSize/2-1));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize/2-1, minChildSize/2-1));
    BOOST_CHECK( minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize/2-1, minChildSize/2+1));
    BOOST_CHECK( minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize/2+1, minChildSize/2-1));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int minChildSize = 6;
    ShouldSplitCriteriaI* minChildSizeCriteria = new MinChildSizeSumCriteria(minChildSize);
    ShouldSplitCriteriaI* clone = minChildSizeCriteria->Clone();
    delete minChildSizeCriteria;

    BOOST_CHECK( clone->ShouldSplit(0, 0.0f, 0, minChildSize/2, minChildSize/2));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize/2-1, minChildSize/2));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize/2, minChildSize/2-1));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize/2-1, minChildSize/2-1));
    BOOST_CHECK( clone->ShouldSplit(0, 0.0f, 0, minChildSize/2-1, minChildSize/2+1));
    BOOST_CHECK( clone->ShouldSplit(0, 0.0f, 0, minChildSize/2+1, minChildSize/2-1));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()