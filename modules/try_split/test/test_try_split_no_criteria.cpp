#include <boost/test/unit_test.hpp>

#include "TrySplitNoCriteria.h"

BOOST_AUTO_TEST_SUITE( TrySplitNoCriteriaTests )

BOOST_AUTO_TEST_CASE(test_ShouldSplit)
{
    TrySplitNoCriteria no_criteria;
     BufferCollection bc;
    BOOST_CHECK(no_criteria.TrySplit(0,0,bc,0, true));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    TrySplitCriteriaI* no_critiera = new TrySplitNoCriteria();
    TrySplitCriteriaI* clone = no_critiera->Clone();
    delete no_critiera;
    BufferCollection bc;
    BOOST_CHECK(clone->TrySplit(0,0,bc,0, true));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()