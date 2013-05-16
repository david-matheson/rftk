#include <boost/test/unit_test.hpp>
#include <time.h>
#include <stdio.h>

#include "TimeLimitCriteria.h"

BOOST_AUTO_TEST_SUITE( TimeLimitCriteriaTests )

BOOST_AUTO_TEST_CASE(test_TrySplit)
{
    const int numberSeconds = 1;
    TimeLimitCriteria timeLimitCriteria(numberSeconds);
    BOOST_CHECK(timeLimitCriteria.TrySplit(0,0));
    time_t startTimeLate = time(NULL);
    while( time(NULL) < startTimeLate+numberSeconds ) {}
    BOOST_CHECK(!timeLimitCriteria.TrySplit(0,0));
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    const int numberSeconds = 1;
    TrySplitCriteriaI* timeLimitCriteria= new TimeLimitCriteria(numberSeconds);
    TrySplitCriteriaI* clone = timeLimitCriteria->Clone();
    delete timeLimitCriteria;
    BOOST_CHECK(clone->TrySplit(0,0));
    time_t startTimeLate = time(NULL);
    while( time(NULL) < startTimeLate+numberSeconds ) {}
    BOOST_CHECK(!clone->TrySplit(0,0));
    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()