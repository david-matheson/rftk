#include <boost/test/unit_test.hpp>

#include "BufferCollection.h"
#include "MatrixBuffer.h"

BOOST_AUTO_TEST_SUITE( BufferCollectionSuite )

template<typename T>
MatrixBufferTemplate<T> CreateExampleMatrix()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    return MatrixBufferTemplate<T>(&data[0], 4, 4);
}

BOOST_AUTO_TEST_CASE(test_AddBuffer)
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb);
    BOOST_CHECK(collection.HasBuffer("double"));
}

BOOST_AUTO_TEST_CASE(test_GetBuffer)
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb);
    MatrixBufferTemplate<double>& mb2 = collection.GetBuffer<MatrixBufferTemplate<double> >("double");

    BOOST_CHECK(mb == mb2);
}

BOOST_AUTO_TEST_CASE(test_GetBuffer_doesnt_copy)
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb);

    MatrixBufferTemplate<double>& mb_get1 = collection.GetBuffer<MatrixBufferTemplate<double> >("double");
    MatrixBufferTemplate<double>& mb_get2 = collection.GetBuffer<MatrixBufferTemplate<double> >("double");

    mb_get1.Set(0,0,100.0);

    bool success = true;
    success &= (&mb_get1 == &mb_get2);
    success &= (mb_get1.Get(0, 0) == mb_get2.Get(0, 0));
    BOOST_CHECK(success);
}

BOOST_AUTO_TEST_CASE(test_AppendBuffer)
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb1 = CreateExampleMatrix<double>();
    MatrixBufferTemplate<double> mb2 = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb1);
    collection.AppendBuffer("double", mb2);

    MatrixBufferTemplate<double> mb3 = CreateExampleMatrix<double>();
    MatrixBufferTemplate<double> mb4 = CreateExampleMatrix<double>();
    mb3.Append(mb4);

    MatrixBufferTemplate<double>& mb_result = collection.GetBuffer<MatrixBufferTemplate<double> >("double");
    BOOST_CHECK(mb_result == mb3);
}

BOOST_AUTO_TEST_SUITE_END()
