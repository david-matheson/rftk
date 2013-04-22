#include <boost/test/unit_test.hpp>

#include "MinImpurityCriteria.h"

BOOST_AUTO_TEST_SUITE( MinImpurityCriteriaTests )

BOOST_AUTO_TEST_CASE(test_ShouldSplit)
{
    const float minImpurity = 0.2;
    MinImpurityCriteria MinImpurityCriteria(minImpurity);
    BOOST_CHECK( MinImpurityCriteria.ShouldSplit(0, minImpurity, 0, 0, 0));
    BOOST_CHECK( MinImpurityCriteria.ShouldSplit(0, minImpurity+0.1, 0, 0, 0));
    BOOST_CHECK( !MinImpurityCriteria.ShouldSplit(0, minImpurity-0.1, 0, 0, 0));

}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const float minImpurity = 0.1;
    ShouldSplitCriteriaI* minImpurityCriteria = new MinImpurityCriteria(minImpurity);
    ShouldSplitCriteriaI* clone = minImpurityCriteria->Clone();
    delete minImpurityCriteria;

    BOOST_CHECK( clone->ShouldSplit(0, minImpurity, 0, 0, 0));
    BOOST_CHECK( clone->ShouldSplit(0, minImpurity+0.1, 0, 0, 0));
    BOOST_CHECK( !clone->ShouldSplit(0, minImpurity-0.1, 0, 0, 0));

    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()