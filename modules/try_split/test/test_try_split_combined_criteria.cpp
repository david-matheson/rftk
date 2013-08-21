#include <boost/test/unit_test.hpp>

#include <vector>

#include "TrySplitCombinedCriteria.h"
#include "MaxDepthCriteria.h"
#include "MinNodeSizeCriteria.h"

BOOST_AUTO_TEST_SUITE( TrySplitCombinedCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    std::vector<TrySplitCriteriaI*> criterias;
    const int maxDepth = 5;
    MaxDepthCriteria maxDepthCriteria(maxDepth);
    criterias.push_back(&maxDepthCriteria);
    const int minDatapoints = 3;
    MinNodeSizeCriteria minNumberDatapoints(minDatapoints);
    criterias.push_back(&minNumberDatapoints);

    TrySplitCombinedCriteria combinedCritiera(criterias);
    BufferCollection bc;

    BOOST_CHECK( !combinedCritiera.TrySplit(maxDepth-1, minDatapoints-1, bc, 0));
    BOOST_CHECK( combinedCritiera.TrySplit(maxDepth-1, minDatapoints, bc, 0));
    BOOST_CHECK( combinedCritiera.TrySplit(maxDepth-1, minDatapoints+1, bc, 0));
    BOOST_CHECK( !combinedCritiera.TrySplit(maxDepth, minDatapoints-1, bc, 0));
    BOOST_CHECK( !combinedCritiera.TrySplit(maxDepth, minDatapoints, bc, 0));
    BOOST_CHECK( !combinedCritiera.TrySplit(maxDepth, minDatapoints+1, bc, 0));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    std::vector<TrySplitCriteriaI*> criterias;
    const int maxDepth = 5;
    TrySplitCriteriaI* maxDepthCriteria = new MaxDepthCriteria(maxDepth);
    criterias.push_back(maxDepthCriteria);
    const int minDatapoints = 3;
    TrySplitCriteriaI* minNumberDatapoints = new MinNodeSizeCriteria(minDatapoints);
    criterias.push_back(minNumberDatapoints);

    TrySplitCriteriaI* combinedCritiera = new TrySplitCombinedCriteria(criterias);
    criterias.clear();
    delete maxDepthCriteria;
    delete minNumberDatapoints;

    TrySplitCriteriaI* clone = combinedCritiera->Clone();
    delete combinedCritiera;
    BufferCollection bc;

    BOOST_CHECK( !clone->TrySplit(maxDepth-1, minDatapoints-1, bc, 0));
    BOOST_CHECK( clone->TrySplit(maxDepth-1, minDatapoints, bc, 0));
    BOOST_CHECK( clone->TrySplit(maxDepth-1, minDatapoints+1, bc, 0));
    BOOST_CHECK( !clone->TrySplit(maxDepth, minDatapoints-1, bc, 0));
    BOOST_CHECK( !clone->TrySplit(maxDepth, minDatapoints, bc, 0));
    BOOST_CHECK( !clone->TrySplit(maxDepth, minDatapoints+1, bc, 0));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()