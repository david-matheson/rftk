#include <boost/test/unit_test.hpp>

#include "UniqueBufferId.h"

BOOST_AUTO_TEST_SUITE( UniqueBufferIdTests )

BOOST_AUTO_TEST_CASE(test_GetId)
{
    UniqueBufferId::Reset();
    BOOST_CHECK_EQUAL( UniqueBufferId::GetId(), 1 );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetId(), 2 );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetId(), 3 );
}

BOOST_AUTO_TEST_CASE(test_GetBufferString)
{
    UniqueBufferId::Reset();
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferId("Try"), "Try1" );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferId("Different"), "Different2" );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferId("Strings"), "Strings3" );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferId("Strings"), "Strings4" );
}

BOOST_AUTO_TEST_SUITE_END()
