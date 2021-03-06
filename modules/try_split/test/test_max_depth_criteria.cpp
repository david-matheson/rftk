#include <boost/test/unit_test.hpp>

#include "MaxDepthCriteria.h"

BOOST_AUTO_TEST_SUITE( MaxDepthCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    const int maxDepth = 5;
    MaxDepthCriteria maxDepthCriteria(maxDepth);
    BufferCollection bc;
    BOOST_CHECK( maxDepthCriteria.TrySplit(maxDepth-1, 0, bc, 0, true));
    BOOST_CHECK( !maxDepthCriteria.TrySplit(maxDepth, 0, bc, 0, true));
    BOOST_CHECK( !maxDepthCriteria.TrySplit(maxDepth+1, 0, bc, 0, true));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int maxDepth = 5;
    TrySplitCriteriaI* maxDepthCriteria = new MaxDepthCriteria(maxDepth);
    TrySplitCriteriaI* clone = maxDepthCriteria->Clone();
    delete maxDepthCriteria;
    BufferCollection bc;
    BOOST_CHECK( clone->TrySplit(maxDepth-1, 0, bc, 0, true));
    BOOST_CHECK( !clone->TrySplit(maxDepth, 0, bc, 0, true));
    BOOST_CHECK( !clone->TrySplit(maxDepth+1, 0, bc, 0, true));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()