#include <boost/test/unit_test.hpp>

#include "ShouldSplitNoCriteria.h"

BOOST_AUTO_TEST_SUITE( ShouldSplitNoCriteriaTests )

BOOST_AUTO_TEST_CASE(test_ShouldSplit)
{
    ShouldSplitNoCriteria no_criteria;
    BOOST_CHECK( no_criteria.ShouldSplit(0, 0.0f, 0, 0, 0));

}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    ShouldSplitCriteriaI* no_critiera = new ShouldSplitNoCriteria();
    ShouldSplitCriteriaI* clone = no_critiera->Clone();
    delete no_critiera;

    BOOST_CHECK( clone->ShouldSplit(0, 0.0f, 0, 0, 0));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()