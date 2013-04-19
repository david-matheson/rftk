#include <boost/test/unit_test.hpp>

#include "MatrixBuffer.h"

BOOST_AUTO_TEST_SUITE( MatrixBufferTests )

template<typename T>
MatrixBufferTemplate<T> CreateExampleMatrix()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    return MatrixBufferTemplate<T>(&data[0], 4, 4);
}

template<typename T>
VectorBufferTemplate<T> CreateExampleVector()
{
    T data[] = {5, 7, 2, 3};
    return VectorBufferTemplate<T>(&data[0], 4);
}

BOOST_AUTO_TEST_CASE(test_SetRow)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    VectorBufferTemplate<double> vb = CreateExampleVector<double>();
    mb.SetRow(2, vb);

    double expected_result_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 5, 7, 2, 3, 12, 13, 14, 15};
    MatrixBufferTemplate<double> expected_result(&expected_result_data[0], 4, 4);

    BOOST_CHECK(mb == expected_result);
}

BOOST_AUTO_TEST_CASE(test_GetMin)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    BOOST_CHECK_EQUAL(mb.GetMin(), 0);
}

BOOST_AUTO_TEST_CASE(test_GetMax)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    BOOST_CHECK_EQUAL(mb.GetMax(), 15);
}

BOOST_AUTO_TEST_CASE(test_SumRow)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    BOOST_CHECK_EQUAL(mb.SumRow(0), 6);
    BOOST_CHECK_EQUAL(mb.SumRow(1), 22);
}

BOOST_AUTO_TEST_CASE(test_NormalizeRow)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    mb.NormalizeRow(1);

    BOOST_CHECK_CLOSE(mb.Get(0,3), 3, 0.001);
    BOOST_CHECK_CLOSE(mb.Get(1,0), 4.0/22, 0.001);
    BOOST_CHECK_CLOSE(mb.Get(1,1), 5.0/22, 0.001);
    BOOST_CHECK_CLOSE(mb.Get(1,2), 6.0/22, 0.001);
    BOOST_CHECK_CLOSE(mb.Get(1,3), 7.0/22, 0.001);
    BOOST_CHECK_CLOSE(mb.Get(2,0), 8, 0.001);
}



BOOST_AUTO_TEST_SUITE_END()