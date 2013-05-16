#include <boost/test/unit_test.hpp>

#include "MaxDepthCriteria.h"

BOOST_AUTO_TEST_SUITE( MaxDepthCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    const int maxDepth = 5;
    MaxDepthCriteria maxDepthCriteria(maxDepth);
    BOOST_CHECK( maxDepthCriteria.TrySplit(maxDepth-1, 0));
    BOOST_CHECK( !maxDepthCriteria.TrySplit(maxDepth, 0));
    BOOST_CHECK( !maxDepthCriteria.TrySplit(maxDepth+1, 0));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int maxDepth = 5;
    TrySplitCriteriaI* maxDepthCriteria = new MaxDepthCriteria(maxDepth);
    TrySplitCriteriaI* clone = maxDepthCriteria->Clone();
    delete maxDepthCriteria;

    BOOST_CHECK( clone->TrySplit(maxDepth-1, 0));
    BOOST_CHECK( !clone->TrySplit(maxDepth, 0));
    BOOST_CHECK( !clone->TrySplit(maxDepth+1, 0));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()