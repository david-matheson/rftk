#include <boost/test/unit_test.hpp>

#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "ScaledDepthDeltaFeature.h"


struct ScaledDepthDeltaFeatureFixture {

    ScaledDepthDeltaFeatureFixture()
    : depth_imgs_key("depth_imgs")
    , scales_key("scales")
    , pixel_indices_key("pixel_indices")
    , indices_key("indices")
    , float_params_key("float_params")
    , int_params_key("int_params")
    , collection()
    , stack()
    {
        float depth_data[] = {2.0, 2.0, 3.0, 1.0,
                              2.0, 1.0, 1.0, 5.0,
                              2.0, 1.0, 1.0, 1.0};
        collection.AddBuffer(depth_imgs_key, Tensor3BufferTemplate<float>(&depth_data[0], 1, 3, 4));
        int pixel_indices_data[] = {0,0,1,
                                    0,2,0, 
                                    0,0,1,
                                    0,0,1};
        collection.AddBuffer(pixel_indices_key, MatrixBufferTemplate<int>(&pixel_indices_data[0], 4, 3));
        int indices_data[] = {0, 1, 2, 3};
        collection.AddBuffer(indices_key, VectorBufferTemplate<int>(&indices_data[0], 4));
        int int_params_data[] = {-1,-1,-1,-1,-1,
                                    -1,-1,-1,-1,-1};
        collection.AddBuffer(int_params_key, MatrixBufferTemplate<int>(&int_params_data[0], 2, 5));
        stack.Push(&collection);
    }

    ~ScaledDepthDeltaFeatureFixture()
    {
    }

    const BufferCollectionKey_t depth_imgs_key;
    const BufferCollectionKey_t scales_key;
    const BufferCollectionKey_t pixel_indices_key;
    const BufferCollectionKey_t indices_key;

    const BufferCollectionKey_t float_params_key;
    const BufferCollectionKey_t int_params_key;

    BufferCollection collection;
    BufferCollectionStack stack;

    typedef BufferTypes<float, int, int, float, int, float, float, int> BufferTypes_t;
    typedef ScaledDepthDeltaFeature< BufferTypes_t > ScaledDepthDeltaFeature_t;
    typedef ScaledDepthDeltaFeatureBinding< BufferTypes_t > ScaledDepthDeltaFeatureBinding_t;
};

BOOST_FIXTURE_TEST_SUITE( ScaledDepthDeltaFeatureTests,  ScaledDepthDeltaFeatureFixture)

BOOST_AUTO_TEST_CASE(test_FeatureValue_default)
{
    float scales_data[] = {1.0, 1.0,
                          1.0, 1.0,
                          1.0, 1.0,
                          1.0, 1.0};
    collection.AddBuffer(scales_key, MatrixBufferTemplate<float>(&scales_data[0], 4, 2));
    float float_params_data[] = {0.0, -1.0, 0.0, 1.0, 0.0,
                                 0.0, 1.0, 2.0, -1.0, -2.0};
    collection.AddBuffer(float_params_key, MatrixBufferTemplate<float>(&float_params_data[0], 2, 5));    

    ScaledDepthDeltaFeature_t feature(  float_params_key, int_params_key,
                                        indices_key, pixel_indices_key,
                                        depth_imgs_key, scales_key);
    ScaledDepthDeltaFeatureBinding_t featureBinding = feature.Bind(stack);

    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 0), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 1), 0.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 2), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 3), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 0), 3.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 1), -1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 2), 3.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 3), 3.0, 0.1);
}

BOOST_AUTO_TEST_CASE(test_FeatureValue_no_scales)
{
    float float_params_data[] = {0.0, -1.0, 0.0, 1.0, 0.0,
                                 0.0, 1.0, 2.0, -1.0, -2.0};
    collection.AddBuffer(float_params_key, MatrixBufferTemplate<float>(&float_params_data[0], 2, 5));    

    ScaledDepthDeltaFeature_t feature(  float_params_key, int_params_key,
                                        indices_key, pixel_indices_key,
                                        depth_imgs_key);
    ScaledDepthDeltaFeatureBinding_t featureBinding = feature.Bind(stack);

    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 0), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 1), 0.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 2), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 3), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 0), 3.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 1), -1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 2), 3.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 3), 3.0, 0.1);
}

BOOST_AUTO_TEST_CASE(test_FeatureValue_scales)
{
    float scales_data[] = {0.5, 2.0,
                          0.5, 2.0,
                          0.5, 2.0,
                          0.5, 2.0};
    collection.AddBuffer(scales_key, MatrixBufferTemplate<float>(&scales_data[0], 4, 2));
    float float_params_data[] = {0.0, -0.5, 0.0, 2.0, 0.0,
                                 0.0, 2.0, 1.0, -2.0, -1.0};
    collection.AddBuffer(float_params_key, MatrixBufferTemplate<float>(&float_params_data[0], 2, 5));    

    ScaledDepthDeltaFeature_t feature(  float_params_key, int_params_key,
                                        indices_key, pixel_indices_key,
                                        depth_imgs_key, scales_key);
    ScaledDepthDeltaFeatureBinding_t featureBinding = feature.Bind(stack);

    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 0), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 1), 0.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 2), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(0, 3), 1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 0), 3.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 1), -1.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 2), 3.0, 0.1);
    BOOST_CHECK_CLOSE(featureBinding.FeatureValue(1, 3), 3.0, 0.1);
}


BOOST_AUTO_TEST_SUITE_END()
