#include <boost/test/unit_test.hpp>

#include <vector>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "AxisAlignedParamsStep.h"


struct AxisAlignedParamsStepFixture {
    AxisAlignedParamsStepFixture()
    : xs_key("xs")
    , number_of_features_key("#features")
    , xs(4,5)
    , numberOfFeaturesBuffers(1)
    , collection()
    , stack()
    {
        collection.AddBuffer(xs_key, xs);
        collection.AddBuffer(number_of_features_key, numberOfFeaturesBuffers);
    }

    ~AxisAlignedParamsStepFixture()
    {
    }

    const BufferCollectionKey_t xs_key;
    const BufferCollectionKey_t number_of_features_key;
    const MatrixBufferTemplate<double> xs;
    const VectorBufferTemplate<int> numberOfFeaturesBuffers;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( AxisAlignedParamsStepTests,  AxisAlignedParamsStepFixture)

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    const int numberOfFeatures = 3;

    VectorBufferTemplate<int>& numberFeaturesBuffer = 
          collection.GetBuffer< VectorBufferTemplate<int> >(number_of_features_key);
    numberFeaturesBuffer.Set(0, numberOfFeatures);
    stack.Push(&collection);

    const AxisAlignedParamsStep<double, int> axisAlignedStep(number_of_features_key, xs_key);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(axisAlignedStep.FloatParamsBufferId));
    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<int> >(axisAlignedStep.IntParamsBufferId));

    boost::mt19937 gen(0);
    axisAlignedStep.ProcessStep(stack, collection, gen);

    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(axisAlignedStep.FloatParamsBufferId));
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<int> >(axisAlignedStep.IntParamsBufferId));    

    MatrixBufferTemplate<double> floatParams = 
            collection.GetBuffer< MatrixBufferTemplate<double> >(axisAlignedStep.FloatParamsBufferId);
    BOOST_CHECK_EQUAL(floatParams.GetM(), numberOfFeatures);
    BOOST_CHECK_EQUAL(floatParams.GetN(), 3);

    MatrixBufferTemplate<int> intParams = 
            collection.GetBuffer< MatrixBufferTemplate<int> >(axisAlignedStep.IntParamsBufferId);
    BOOST_CHECK_EQUAL(intParams.GetM(), numberOfFeatures);
    BOOST_CHECK_EQUAL(intParams.GetN(), 3);

    std::vector<bool> dimensionsUsed(numberOfFeatures);
    for(int i=0; i<numberOfFeatures; i++)
    {
        const int featureType = intParams.Get(i,0);
        BOOST_CHECK(featureType == MATRIX_FEATURES);
        const int numberOfDimensions = intParams.Get(i,1);
        BOOST_CHECK_EQUAL(numberOfDimensions, 1);
        const int dimension = intParams.Get(i,2);
        BOOST_CHECK(dimension >=0 && dimension < 5);
        BOOST_CHECK(!dimensionsUsed[dimension]);
        dimensionsUsed[dimension] = true;

        const double weight = floatParams.Get(i,2);
        BOOST_CHECK_CLOSE(1.0, weight, 0.001);
    }
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_all_dimensions)
{
    const int numberOfFeaturesRequested = 10;
    const int numberOfFeaturesActual = 5;
    VectorBufferTemplate<int>& numberFeaturesBuffer = 
          collection.GetBuffer< VectorBufferTemplate<int> >(number_of_features_key);
    numberFeaturesBuffer.Set(0, numberOfFeaturesRequested);
    stack.Push(&collection);

    const AxisAlignedParamsStep<double, int> axisAlignedStep(number_of_features_key, xs_key);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(axisAlignedStep.FloatParamsBufferId));
    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<int> >(axisAlignedStep.IntParamsBufferId));

    boost::mt19937 gen(0);
    axisAlignedStep.ProcessStep(stack, collection, gen);

    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(axisAlignedStep.FloatParamsBufferId));
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<int> >(axisAlignedStep.IntParamsBufferId));    

    MatrixBufferTemplate<double> floatParams = 
            collection.GetBuffer< MatrixBufferTemplate<double> >(axisAlignedStep.FloatParamsBufferId);
    BOOST_CHECK_EQUAL(floatParams.GetM(), numberOfFeaturesActual);
    BOOST_CHECK_EQUAL(floatParams.GetN(), 3);

    MatrixBufferTemplate<int> intParams = 
            collection.GetBuffer< MatrixBufferTemplate<int> >(axisAlignedStep.IntParamsBufferId);
    BOOST_CHECK_EQUAL(intParams.GetM(), numberOfFeaturesActual);
    BOOST_CHECK_EQUAL(intParams.GetN(), 3);

    std::vector<bool> dimensionsUsed(numberOfFeaturesActual);
    for(int i=0; i<numberOfFeaturesActual; i++)
    {
        const int featureType = intParams.Get(i,0);
        BOOST_CHECK(featureType == MATRIX_FEATURES);
        const int numberOfDimensions = intParams.Get(i,1);
        BOOST_CHECK_EQUAL(numberOfDimensions, 1);
        const int dimension = intParams.Get(i,2);
        BOOST_CHECK(dimension >=0 && dimension < 5);
        BOOST_CHECK(!dimensionsUsed[dimension]);
        dimensionsUsed[dimension] = true;

        const double weight = floatParams.Get(i,2);
        BOOST_CHECK_CLOSE(1.0, weight, 0.001);
    }
}

BOOST_AUTO_TEST_SUITE_END()
