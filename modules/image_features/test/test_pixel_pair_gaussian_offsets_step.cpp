#include <boost/test/unit_test.hpp>

#include <vector>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "PixelPairGaussianOffsetsStep.h"


struct PixelPairGaussianOffsetsStepFixture {
    PixelPairGaussianOffsetsStepFixture()
    : number_of_features_key("#features")
    , numberOfFeaturesBuffers(1)
    , collection()
    , stack()
    {
        collection.AddBuffer(number_of_features_key, numberOfFeaturesBuffers);
    }

    ~PixelPairGaussianOffsetsStepFixture()
    {
    }

    const BufferCollectionKey_t number_of_features_key;
    const VectorBufferTemplate<int> numberOfFeaturesBuffers;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( PixelPairGaussianOffsetsStepTests,  PixelPairGaussianOffsetsStepFixture)

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    const int numberOfFeatures = 100000;
    const double offsets[] = {200, 100, 500, 800};

    VectorBufferTemplate<int>& numberFeaturesBuffer = 
          collection.GetBuffer< VectorBufferTemplate<int> >(number_of_features_key);
    numberFeaturesBuffer.Set(0, numberOfFeatures);
    stack.Push(&collection);

    const PixelPairGaussianOffsetsStep<double, int> pixelPairOffsetStep(number_of_features_key, offsets[0], offsets[1], offsets[2], offsets[3]);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(pixelPairOffsetStep.FloatParamsBufferId));
    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<int> >(pixelPairOffsetStep.IntParamsBufferId));

    boost::mt19937 gen(0);
    pixelPairOffsetStep.ProcessStep(stack, collection, gen);

    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(pixelPairOffsetStep.FloatParamsBufferId));
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<int> >(pixelPairOffsetStep.IntParamsBufferId));    

    MatrixBufferTemplate<double> floatParams = 
            collection.GetBuffer< MatrixBufferTemplate<double> >(pixelPairOffsetStep.FloatParamsBufferId);
    BOOST_CHECK_EQUAL(floatParams.GetM(), numberOfFeatures);
    BOOST_CHECK_EQUAL(floatParams.GetN(), 5);

    MatrixBufferTemplate<int> intParams = 
            collection.GetBuffer< MatrixBufferTemplate<int> >(pixelPairOffsetStep.IntParamsBufferId);
    BOOST_CHECK_EQUAL(intParams.GetM(), numberOfFeatures);
    BOOST_CHECK_EQUAL(intParams.GetN(), 5);

    std::vector<double> values(4);
    std::vector<double> values_2(4);
    for(int i=0; i<numberOfFeatures; i++)
    {
        for(int d=1; d<5; d++)
        {
            values[d-1] += floatParams.Get(i, d);
            values_2[d-1] += floatParams.Get(i, d)*floatParams.Get(i, d);
        }
    }

    for(int d=0; d<4; d++)
    {
        BOOST_REQUIRE_LE(values[d] / double(numberOfFeatures), 10.0);
        BOOST_CHECK_CLOSE(sqrt(values_2[d] / double(numberOfFeatures)), offsets[d], 10.0);
    }
}


BOOST_AUTO_TEST_SUITE_END()
