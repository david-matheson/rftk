#include <boost/test/unit_test.hpp>

#include "MatrixBuffer.h"
#include "FeatureEqualQuantized.h"


BOOST_AUTO_TEST_SUITE( FeatureEqualQuantizedTests )

typedef SinglePrecisionBufferTypes FeatureEqualQuantizedTests_BufferTypes_t;

BOOST_AUTO_TEST_CASE(test_IsEqual)
{
    float float_1_data[] = {0, 1, 2, 3, 4, 
                            5, 6, 7, 8, 9, 
                            0, 1, 2, 3, 4,};
    MatrixBufferTemplate<float> float_1(&float_1_data[0], 3, 5);

    int int_1_data[] = {22, 2, 5, 33, 4, 
                            5, 6, 73, 8, 9, 
                            22, 2, 5, 33, 4};
    MatrixBufferTemplate<int> int_1(&int_1_data[0], 3, 5);

    float float_2_data[] = {0, 1, 2, 3, 4, 
                            0, 1, 2, 3, 4,
                            5, 6, 7, 8, 9  };
    MatrixBufferTemplate<float> float_2(&float_2_data[0], 3, 5);

    int int_2_data[] = {22, 2, 5, 33, 4, 
                            22, 2, 5, 33, 4,
                            5, 6, 73, 8, 9};
    MatrixBufferTemplate<int> int_2(&int_2_data[0], 3, 5);


    FeatureEqualQuantized<FeatureEqualQuantizedTests_BufferTypes_t> featureEqualQuantized(1.0f);
    BOOST_CHECK(featureEqualQuantized.IsEqual(float_1, int_1, 0, float_2, int_2, 0));
    BOOST_CHECK(featureEqualQuantized.IsEqual(float_1, int_1, 0, float_2, int_2, 1));
    BOOST_CHECK(!featureEqualQuantized.IsEqual(float_1, int_1, 0, float_2, int_2, 2));
    BOOST_CHECK(!featureEqualQuantized.IsEqual(float_1, int_1, 1, float_2, int_2, 0));
    BOOST_CHECK(!featureEqualQuantized.IsEqual(float_1, int_1, 1, float_2, int_2, 1));
    BOOST_CHECK(featureEqualQuantized.IsEqual(float_1, int_1, 1, float_2, int_2, 2));
    BOOST_CHECK(featureEqualQuantized.IsEqual(float_1, int_1, 2, float_2, int_2, 0));
    BOOST_CHECK(featureEqualQuantized.IsEqual(float_1, int_1, 2, float_2, int_2, 1));
    BOOST_CHECK(!featureEqualQuantized.IsEqual(float_1, int_1, 2, float_2, int_2, 2));
}

BOOST_AUTO_TEST_CASE(test_IsEqual_different_precision)
{
    float float_1_data[] = {0.11, 1, 2, 3, 4};
    MatrixBufferTemplate<float> float_1(&float_1_data[0], 1, 5);

    int int_1_data[] = {22, 2, 5, 33, 4};
    MatrixBufferTemplate<int> int_1(&int_1_data[0], 1, 5);

    float float_2_data[] = {0.12, 1, 2, 3, 4 };
    MatrixBufferTemplate<float> float_2(&float_2_data[0], 1, 5);

    int int_2_data[] = {22, 2, 5, 33, 4};
    MatrixBufferTemplate<int> int_2(&int_2_data[0], 1, 5);

    FeatureEqualQuantized<FeatureEqualQuantizedTests_BufferTypes_t> featureEqualQuantized_1(1.0f);
    BOOST_CHECK(featureEqualQuantized_1.IsEqual(float_1, int_1, 0, float_2, int_2, 0));

    FeatureEqualQuantized<FeatureEqualQuantizedTests_BufferTypes_t> featureEqualQuantized_10(10.0f);
    BOOST_CHECK(featureEqualQuantized_10.IsEqual(float_1, int_1, 0, float_2, int_2, 0));

    FeatureEqualQuantized<FeatureEqualQuantizedTests_BufferTypes_t> featureEqualQuantized_100(100.0f);
    BOOST_CHECK(!featureEqualQuantized_100.IsEqual(float_1, int_1, 0, float_2, int_2, 0));
}


BOOST_AUTO_TEST_SUITE_END()