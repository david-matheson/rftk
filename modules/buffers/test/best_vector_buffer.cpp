#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"

BOOST_AUTO_TEST_SUITE( VectorBufferTests )

template<typename T>
VectorBufferTemplate<T> CreateExampleVector()
{
    T data[] = {0, 1, 2, 3, 4};
    return VectorBufferTemplate<T>(&data[0], 5);
}

BOOST_AUTO_TEST_CASE(test_Sum)
{
    VectorBufferTemplate<double> v = CreateExampleVector<double>();
    BOOST_CHECK_EQUAL(v.Sum(), 10);
}

BOOST_AUTO_TEST_CASE(test_Normalize)
{
    VectorBufferTemplate<double> v = CreateExampleVector<double>();
    v.Normalize();
    BOOST_CHECK_EQUAL(v.Sum(), 1);

    double expected_result_data[] = {0, 0.1, 0.2, 0.3, 0.4};
    VectorBufferTemplate<double> expected_result(&expected_result_data[0], 5);
    BOOST_CHECK(v == expected_result);
}

BOOST_AUTO_TEST_CASE(test_Normalized)
{
    VectorBufferTemplate<double> v = CreateExampleVector<double>();
    VectorBufferTemplate<double> v_normalized = v.Normalized();

    BOOST_CHECK_EQUAL(v_normalized.Sum(), 1);
    BOOST_CHECK(!(v == v_normalized));
    BOOST_CHECK_EQUAL(v.Sum(), 10);

    double expected_result_data[] = {0, 0.1, 0.2, 0.3, 0.4};
    VectorBufferTemplate<double> expected_result(&expected_result_data[0], 5);
    BOOST_CHECK(v_normalized == expected_result);
}


BOOST_AUTO_TEST_SUITE_END()