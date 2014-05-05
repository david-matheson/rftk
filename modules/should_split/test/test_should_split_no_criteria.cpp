#include <boost/test/unit_test.hpp>

#include "ShouldSplitNoCriteria.h"

BOOST_AUTO_TEST_SUITE( ShouldSplitNoCriteriaTests )

BOOST_AUTO_TEST_CASE(test_ShouldSplit)
{
    ShouldSplitNoCriteria no_criteria;
    BufferCollection bc;
    BOOST_CHECK( no_criteria.ShouldSplit(0, 0.0f, 0, 0, 0, bc, 0, true));

}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    ShouldSplitCriteriaI* no_critiera = new ShouldSplitNoCriteria();
    ShouldSplitCriteriaI* clone = no_critiera->Clone();
    delete no_critiera;

    BufferCollection bc;
    BOOST_CHECK( clone->ShouldSplit(0, 0.0f, 0, 0, 0, bc, 0, true));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()