#include <string>

#include "TestHarness.h"

#include "BufferCollection.h"
#include "MatrixBuffer.h"

template<typename T>
MatrixBufferTemplate<T> CreateExampleMatrix()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    return MatrixBufferTemplate<T>(&data[0], 4, 4);
}

bool test_AddBuffer()
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb);
    return collection.HasBuffer("double");
}

bool test_GetBuffer()
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb);
    MatrixBufferTemplate<double>& mb2 = collection.GetBuffer<MatrixBufferTemplate<double> >("double");

    return mb == mb2;
}

bool test_GetBuffer_doesnt_copy()
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
    return success;
}

bool test_AppendBuffer()
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
    return mb_result == mb3;
}

int main() {
    bool all_success = true;


    RUN_TEST(test_AddBuffer);
    RUN_TEST(test_GetBuffer);
    RUN_TEST(test_GetBuffer_doesnt_copy);
    RUN_TEST(test_AppendBuffer);

    return !all_success;
}
