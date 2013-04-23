#include <boost/test/unit_test.hpp>

#include "Tensor3Buffer.h"

BOOST_AUTO_TEST_SUITE( Tensor3BufferTests )

template<typename T>
Tensor3BufferTemplate<T> CreateExampleTensor3()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    return Tensor3BufferTemplate<T>(&data[0], 3, 2, 4);
}

template<typename T>
VectorBufferTemplate<T> CreateExampleVector()
{
    T data[] = {5, 7, 2, 3};
    return VectorBufferTemplate<T>(&data[0], 4);
}

BOOST_AUTO_TEST_CASE(test_Tensor3_SetRow)
{
    Tensor3BufferTemplate<int> tb = CreateExampleTensor3<int>();
    VectorBufferTemplate<int> vb = CreateExampleVector<int>();
    tb.SetRow(2, 1, vb);

    int expected_result_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 5, 7, 2, 3};
    Tensor3BufferTemplate<int> expected_result(&expected_result_data[0], 3, 2, 4);

    BOOST_CHECK(tb == expected_result);
}

BOOST_AUTO_TEST_CASE(test_Tensor3_SliceRow)
{
    Tensor3BufferTemplate<int> tb = CreateExampleTensor3<int>();
    VectorBufferTemplate<int> vb = tb.SliceRow(2, 0);

    int expected_result_data[] = { 16, 17, 18, 19 };
    VectorBufferTemplate<int> expected_result(&expected_result_data[0], 4);

    BOOST_CHECK(vb == expected_result);
}

BOOST_AUTO_TEST_CASE(test_Tensor3_SumRow)
{
    Tensor3BufferTemplate<double> tb = CreateExampleTensor3<double>();
    BOOST_CHECK_EQUAL(tb.SumRow(0,0), 6);
    BOOST_CHECK_EQUAL(tb.SumRow(1, 0), 38);
    BOOST_CHECK_EQUAL(tb.SumRow(1, 1), 54);
}

BOOST_AUTO_TEST_CASE(test_Tensor3_NormalizeRow)
{
    Tensor3BufferTemplate<double> tb = CreateExampleTensor3<double>();
    tb.NormalizeRow(1,1);

    BOOST_CHECK_CLOSE(tb.Get(1,0,3), 11, 0.001);
    BOOST_CHECK_CLOSE(tb.Get(1,1,0), 12.0/54, 0.001);
    BOOST_CHECK_CLOSE(tb.Get(1,1,1), 13.0/54, 0.001);
    BOOST_CHECK_CLOSE(tb.Get(1,1,2), 14.0/54, 0.001);
    BOOST_CHECK_CLOSE(tb.Get(1,1,3), 15.0/54, 0.001);
    BOOST_CHECK_CLOSE(tb.Get(2,0,0), 16, 0.001);
}

BOOST_AUTO_TEST_SUITE_END()