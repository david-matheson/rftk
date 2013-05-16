#include <boost/test/unit_test.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "SetBufferStep.h"

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


struct SetBufferStepFixture {
    SetBufferStepFixture()
    : buffer1(Create4x4ExampleMatrix1<double>())
    , buffer2(Create4x4ExampleMatrix2<double>())
    , collection()
    , stack()
    {
    }

    ~SetBufferStepFixture()
    {
    }

    const MatrixBufferTemplate<double> buffer1;
    const MatrixBufferTemplate<double> buffer2;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( SetBufferStepTests, SetBufferStepFixture )

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    const SetBufferStep< MatrixBufferTemplate<double> > setBufferStep(buffer1, WHEN_NEW);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId));
    boost::mt19937 gen(0);
    setBufferStep.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId));
    BOOST_CHECK(collection.GetBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId) == buffer1);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_when_new)
{
    const SetBufferStep< MatrixBufferTemplate<double> > setBufferStep(buffer1, WHEN_NEW);
    collection.AddBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId, buffer2);
    BOOST_CHECK(collection.GetBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId) == buffer2);
    boost::mt19937 gen(0);
    setBufferStep.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.GetBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId) == buffer2);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_every_process)
{
    const SetBufferStep< MatrixBufferTemplate<double> > setBufferStep(buffer1, EVERY_PROCESS);
    collection.AddBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId, buffer2);
    BOOST_CHECK(collection.GetBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId) == buffer2);
    boost::mt19937 gen(0);
    setBufferStep.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.GetBuffer< MatrixBufferTemplate<double> >(setBufferStep.OutputBufferId) == buffer1);
}

BOOST_AUTO_TEST_SUITE_END()
