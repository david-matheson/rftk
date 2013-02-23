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

    double const* begin = mb.GetRowPtrUnsafe(0);
    return std::equal(begin, begin + mb.GetM()*mb.GetN(), mb2.GetRowPtrUnsafe(0));
}

bool test_GetBuffer_doesnt_copy()
{
    BufferCollection collection;
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();
    collection.AddBuffer("double", mb);
    MatrixBufferTemplate<double>& mb2 = collection.GetBuffer<MatrixBufferTemplate<double> >("double");

    mb.Set(0,0,100.0);

    bool success = true;
    success &= (mb2.Get(0, 0) == mb.Get(0,0));
    success &= (&mb2 == &mb);
    return success;
}

int main() {
    bool all_success = true;


    RUN_TEST(test_AddBuffer);
    RUN_TEST(test_GetBuffer);
    RUN_TEST(test_GetBuffer_doesnt_copy);

    return !all_success;
}
