#include <boost/test/unit_test.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "BufferTypes.h"
#include "FeatureRangeStep.h"


template<typename T>
MatrixBufferTemplate<T> CreateFeatureValuesBuffer()
{
    T data[] = {0.2, 0.4, 0.6, 0.5,
                 -1.5, -2.5, -30, -0.3,
                  -5, 0, 0.1, 5.1};
    return MatrixBufferTemplate<T>(&data[0], 3, 4);
}

template<typename T>
MatrixBufferTemplate<T> CreateFeatureValuesBufferTranspose()
{
    T data[] = {0.2, -1.5, -5, 
                0.4, -2.5, 0, 
                0.6, -30, 0.1, 
                0.5, -0.3, 5.1};
    return MatrixBufferTemplate<T>(&data[0], 4, 3);
}


struct FeatureRangeStepFixture {
    FeatureRangeStepFixture()
    : featureValuesBufferId("featureValuesBufferId")
    , featureValuesTransposeBufferId("featureValuesTransposeBufferId")
    , featureValuesBuffer(CreateFeatureValuesBuffer<float>())
    , featureValuesBufferTranspose(CreateFeatureValuesBufferTranspose<float>())
    , collection()
    , stack()
    {
        collection.AddBuffer< MatrixBufferTemplate<float> >(featureValuesBufferId, featureValuesBuffer);
        collection.AddBuffer< MatrixBufferTemplate<float> >(featureValuesTransposeBufferId, featureValuesBufferTranspose);
        stack.Push(&collection);
    }

    ~FeatureRangeStepFixture()
    {
    }

    const BufferId featureValuesBufferId;
    const BufferId featureValuesTransposeBufferId;
    const MatrixBufferTemplate<float> featureValuesBuffer;
    const MatrixBufferTemplate<float> featureValuesBufferTranspose;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( FeatureRangeStepTests, FeatureRangeStepFixture )

BOOST_AUTO_TEST_CASE(test_ProcessStep_FEATURES_BY_DATAPOINTS)
{
    const FeatureRangeStep< SinglePrecisionBufferTypes > featureRange(featureValuesBufferId, FEATURES_BY_DATAPOINTS);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    boost::mt19937 gen(0);
    featureRange.ProcessStep(stack, collection, gen, collection, 0);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    const MatrixBufferTemplate<float>& fvBuffer = collection.GetBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId);
    BOOST_CHECK_EQUAL(fvBuffer.GetM(), 3);
    BOOST_CHECK_EQUAL(fvBuffer.GetN(), 2);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,0), 0.2, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,1), 0.6, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,0), -30, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,1), -0.3, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,0), -5, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,1), 5.1, 0.001);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_DATAPOINTS_BY_FEATURES)
{
    const FeatureRangeStep< SinglePrecisionBufferTypes > featureRange(featureValuesBufferId, DATAPOINTS_BY_FEATURES);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    boost::mt19937 gen(0);
    featureRange.ProcessStep(stack, collection, gen, collection, 0);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    const MatrixBufferTemplate<float>& fvBuffer = collection.GetBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId);
    BOOST_CHECK_EQUAL(fvBuffer.GetM(), 4);
    BOOST_CHECK_EQUAL(fvBuffer.GetN(), 2);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,0), -5, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,1), 0.2, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,0), -2.5, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,1), 0.4, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,0), -30, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,1), 0.6, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(3,0), -0.3, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(3,1), 5.1, 0.001);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_Transpose_FEATURES_BY_DATAPOINTS)
{
    const FeatureRangeStep< SinglePrecisionBufferTypes > featureRange(featureValuesTransposeBufferId, FEATURES_BY_DATAPOINTS);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    boost::mt19937 gen(0);
    featureRange.ProcessStep(stack, collection, gen, collection, 0);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    const MatrixBufferTemplate<float>& fvBuffer = collection.GetBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId);
    BOOST_CHECK_EQUAL(fvBuffer.GetM(), 4);
    BOOST_CHECK_EQUAL(fvBuffer.GetN(), 2);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,0), -5, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,1), 0.2, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,0), -2.5, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,1), 0.4, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,0), -30, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,1), 0.6, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(3,0), -0.3, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(3,1), 5.1, 0.001);
}


BOOST_AUTO_TEST_CASE(test_ProcessStep_Transpose_DATAPOINTS_BY_FEATURES)
{
    const FeatureRangeStep< SinglePrecisionBufferTypes > featureRange(featureValuesTransposeBufferId, DATAPOINTS_BY_FEATURES);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    boost::mt19937 gen(0);
    featureRange.ProcessStep(stack, collection, gen, collection, 0);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId));
    const MatrixBufferTemplate<float>& fvBuffer = collection.GetBuffer< MatrixBufferTemplate<float> >(featureRange.FeatureRangeMinMaxBufferId);
    BOOST_CHECK_EQUAL(fvBuffer.GetM(), 3);
    BOOST_CHECK_EQUAL(fvBuffer.GetN(), 2);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,0), 0.2, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(0,1), 0.6, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,0), -30, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(1,1), -0.3, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,0), -5, 0.001);
    BOOST_CHECK_CLOSE(fvBuffer.Get(2,1), 5.1, 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
