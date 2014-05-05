#include <boost/test/unit_test.hpp>

#include <vector>

#include "ShouldSplitCombinedCriteria.h"
#include "MinChildSizeCriteria.h"
#include "MinImpurityCriteria.h"

BOOST_AUTO_TEST_SUITE( ShouldSplitCombinedCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    std::vector<ShouldSplitCriteriaI*> criterias;
    const int minChildSize = 5;
    MinChildSizeCriteria minChildSizeCriteria(minChildSize);
    criterias.push_back(&minChildSizeCriteria);
    const float minImpurity = 0.2;
    MinImpurityCriteria minImpurityCriteria(minImpurity);
    criterias.push_back(&minImpurityCriteria);

    ShouldSplitCombinedCriteria combinedCriteria(criterias);

    BufferCollection bc;
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity, 0, minChildSize, minChildSize, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity, 0, minChildSize-1, minChildSize, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity, 0, minChildSize, minChildSize-1, bc, 0, true));
    BOOST_CHECK( combinedCriteria.ShouldSplit(0, minImpurity+0.1, 0, minChildSize, minChildSize, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity+0.1, 0, minChildSize-1, minChildSize, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity+0.1, 0, minChildSize, minChildSize-1, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity-0.1, 0, minChildSize, minChildSize, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity-0.1, 0, minChildSize-1, minChildSize, bc, 0, true));
    BOOST_CHECK( !combinedCriteria.ShouldSplit(0, minImpurity-0.1, 0, minChildSize, minChildSize-1, bc, 0, true));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    std::vector<ShouldSplitCriteriaI*> criterias;
    const int minChildSize = 5;
    ShouldSplitCriteriaI* minChildSizeCriteria = new MinChildSizeCriteria(minChildSize);
    criterias.push_back(minChildSizeCriteria);
    const float minImpurity = 0.2;
    ShouldSplitCriteriaI* minImpurityCriteria = new MinImpurityCriteria(minImpurity);
    criterias.push_back(minImpurityCriteria);

    ShouldSplitCriteriaI* combinedCriteria = new ShouldSplitCombinedCriteria(criterias);
    criterias.clear();
    delete minChildSizeCriteria;
    delete minImpurityCriteria;

    ShouldSplitCriteriaI* clone = combinedCriteria->Clone();
    delete combinedCriteria;

    BufferCollection bc;
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity, 0, minChildSize, minChildSize, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity, 0, minChildSize-1, minChildSize, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity, 0, minChildSize, minChildSize-1, bc, 0, true));
    BOOST_CHECK( clone->ShouldSplit(0, minImpurity+0.1, 0, minChildSize, minChildSize, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity+0.1, 0, minChildSize-1, minChildSize, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity+0.1, 0, minChildSize, minChildSize-1, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity-0.1, 0, minChildSize, minChildSize, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity-0.1, 0, minChildSize-1, minChildSize, bc, 0, true));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity-0.1, 0, minChildSize, minChildSize-1, bc, 0, true));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()