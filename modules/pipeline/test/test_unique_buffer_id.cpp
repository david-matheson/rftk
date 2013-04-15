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
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferString("Try"), "Try1" );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferString("Different"), "Different2" );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferString("Strings"), "Strings3" );
    BOOST_CHECK_EQUAL( UniqueBufferId::GetBufferString("Strings"), "Strings4" );
}

BOOST_AUTO_TEST_SUITE_END()
