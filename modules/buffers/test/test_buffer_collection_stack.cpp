#include <boost/test/unit_test.hpp>

#include "BufferCollectionStack.h"
#include "BufferCollection.h"
#include "MatrixBuffer.h"


template<typename T>
MatrixBufferTemplate<T> Create4x4ExampleMatrix1()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    return MatrixBufferTemplate<T>(&data[0], 4, 4);
}

template<typename T>
MatrixBufferTemplate<T> Create4x4ExampleMatrix2()
{
    T data[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    return MatrixBufferTemplate<T>(&data[0], 4, 4);
}

struct Fixture {
    Fixture() 
    : double4x4_1(Create4x4ExampleMatrix1<double>())
    , double4x4_2(Create4x4ExampleMatrix2<double>())
    , float4x4_1(Create4x4ExampleMatrix1<float>())
    , collection_1()
    , collection_2()
    {
        collection_1.AddBuffer("double4x4_1", double4x4_1);
        collection_2.AddBuffer("float4x4_1", float4x4_1);
    }

    ~Fixture()
    {
    }

    const MatrixBufferTemplate<double> double4x4_1;
    const MatrixBufferTemplate<double> double4x4_2;
    const MatrixBufferTemplate<float> float4x4_1;
    BufferCollection collection_1;
    BufferCollection collection_2;
};

BOOST_FIXTURE_TEST_SUITE( BufferCollectionStackSuite, Fixture )

BOOST_AUTO_TEST_CASE(test_HasBuffer)
{
    BufferCollectionStack stack;
    stack.Push(&collection_1);

    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<double> >("other"));
}

BOOST_AUTO_TEST_CASE(test_GetBuffer)
{
    BufferCollectionStack stack;
    stack.Push(&collection_1);
    const MatrixBufferTemplate<double>& mb = 
        stack.GetBuffer< MatrixBufferTemplate<double> >("double4x4_1");
    BOOST_CHECK(mb == double4x4_1);
}

BOOST_AUTO_TEST_CASE(test_Push_and_Pop)
{
    BufferCollectionStack stack;
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<float> >("float4x4_1"));
    
    stack.Push(&collection_1);
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<float> >("float4x4_1"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("double4x4_1") == double4x4_1);
    
    stack.Push(&collection_2);
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<float> >("float4x4_1"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("double4x4_1") == double4x4_1);
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<float> >("float4x4_1") == float4x4_1);
    
    stack.Pop();
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<float> >("float4x4_1"));

    stack.Pop();
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<float> >("float4x4_1"));
}

BOOST_AUTO_TEST_CASE(test_Push_Override_Buffer)
{
    BufferCollectionStack stack;
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    
    stack.Push(&collection_1);
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("double4x4_1") == double4x4_1);  

    collection_2.AddBuffer("double4x4_1", double4x4_2);
    stack.Push(&collection_2);
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("double4x4_1") == double4x4_2);     

    stack.Pop();
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("double4x4_1") == double4x4_1);

    stack.Pop();
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<double> >("double4x4_1"));
}

BOOST_AUTO_TEST_CASE(test_Same_Name_Different_Types)
{
    MatrixBufferTemplate<double> conflict_double = Create4x4ExampleMatrix1<double>();
    collection_1.AddBuffer("conflict", conflict_double);
    MatrixBufferTemplate<float> conflict_float = Create4x4ExampleMatrix2<float>();
    collection_2.AddBuffer("conflict", conflict_float);

    BufferCollectionStack stack;
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<double> >("conflict"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<float> >("conflict"));

    stack.Push(&collection_1);
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("conflict"));
    BOOST_CHECK(!stack.HasBuffer< MatrixBufferTemplate<float> >("conflict"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("conflict") == conflict_double);  

    stack.Push(&collection_2);
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<double> >("conflict"));
    BOOST_CHECK(stack.HasBuffer< MatrixBufferTemplate<float> >("conflict"));
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<double> >("conflict") == conflict_double);  
    BOOST_CHECK(stack.GetBuffer< MatrixBufferTemplate<float> >("conflict") == conflict_float);  
}

BOOST_AUTO_TEST_SUITE_END()
