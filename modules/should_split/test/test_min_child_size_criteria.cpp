#include <boost/test/unit_test.hpp>

#include "MinChildSizeCriteria.h"

BOOST_AUTO_TEST_SUITE( MinChildSizeCriteriaTests )

BOOST_AUTO_TEST_CASE(test_ShouldSplit)
{
    const int minChildSize = 5;
    MinChildSizeCriteria minChildSizeCriteria(minChildSize);
    BufferCollection bc;
    BOOST_CHECK( minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize, minChildSize, bc, 0));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize-1, minChildSize, bc, 0));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize, minChildSize-1, bc, 0));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize-1, minChildSize-1, bc, 0));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize-1, minChildSize+2, bc, 0));
    BOOST_CHECK( !minChildSizeCriteria.ShouldSplit(0, 0.0f, 0, minChildSize+2, minChildSize-1, bc, 0));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int minChildSize = 5;
    ShouldSplitCriteriaI* minChildSizeCriteria = new MinChildSizeCriteria(minChildSize);
    ShouldSplitCriteriaI* clone = minChildSizeCriteria->Clone();
    delete minChildSizeCriteria;
    BufferCollection bc;
    BOOST_CHECK( clone->ShouldSplit(0, 0.0f, 0, minChildSize, minChildSize, bc, 0));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize-1, minChildSize, bc, 0));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize, minChildSize-1, bc, 0));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize-1, minChildSize-1, bc, 0));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize-1, minChildSize+2, bc, 0));
    BOOST_CHECK( !clone->ShouldSplit(0, 0.0f, 0, minChildSize+2, minChildSize-1, bc, 0));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()