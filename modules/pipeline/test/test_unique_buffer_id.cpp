#include <boost/test/unit_test.hpp>

#include "UniqueBufferId.h"

BOOST_AUTO_TEST_SUITE( UniqueBufferIdTests )

BOOST_AUTO_TEST_CASE(test_GetId)
{
    Reset();
    BOOST_CHECK_EQUAL( GetId(), 1 );
    BOOST_CHECK_EQUAL( GetId(), 2 );
    BOOST_CHECK_EQUAL( GetId(), 3 );
}

BOOST_AUTO_TEST_CASE(test_GetBufferString)
{
    Reset();
    BOOST_CHECK_EQUAL( GetBufferId("Try"), "Try1" );
    BOOST_CHECK_EQUAL( GetBufferId("Different"), "Different2" );
    BOOST_CHECK_EQUAL( GetBufferId("Strings"), "Strings3" );
    BOOST_CHECK_EQUAL( GetBufferId("Strings"), "Strings4" );
}

BOOST_AUTO_TEST_SUITE_END()
