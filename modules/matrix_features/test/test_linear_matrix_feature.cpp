#include <boost/test/unit_test.hpp>

#include <vector>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "LinearMatrixFeature.h"
#include "FeatureExtractorStep.h"


template<typename T>
MatrixBufferTemplate<T> Create4x5ExampleMatrix()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    return MatrixBufferTemplate<T>(&data[0], 4, 5);
}

template<typename T>
VectorBufferTemplate<T> CreateRange5()
{
    T data[] = {0, 1, 2, 3};
    return VectorBufferTemplate<T>(&data[0], 4);
}

struct LinearMatrixFeatureFixture {

    LinearMatrixFeatureFixture()
    : float_params_key("float_params")
    , int_params_key("int_params")
    , xs_key("xs")
    , indices_key("indices")
    , xs(Create4x5ExampleMatrix<double>())
    , indices(CreateRange5<int>())
    , collection()
    , stack()
    {
        collection.AddBuffer(xs_key, xs);
        collection.AddBuffer(indices_key, indices);
        stack.Push(&collection);
    }

    ~LinearMatrixFeatureFixture()
    {
    }

    const BufferCollectionKey_t float_params_key;
    const BufferCollectionKey_t int_params_key;
    const BufferCollectionKey_t xs_key;
    const BufferCollectionKey_t indices_key;
    const MatrixBufferTemplate<double> xs;
    const VectorBufferTemplate<int> indices;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( LinearMatrixFeatureTests,  LinearMatrixFeatureFixture)

BOOST_AUTO_TEST_CASE(test_FeatureValue_axis_aligned)
{
    double float_params_data[] = {0, 0, 1,
                                  0, 0, 1};
    MatrixBufferTemplate<double> float_params(&float_params_data[0], 2, 3);
    collection.AddBuffer< MatrixBufferTemplate<double> >(float_params_key, float_params);

    int int_params_data[] = {MATRIX_FEATURES, 1, 0,
                             MATRIX_FEATURES, 1, 3};
    MatrixBufferTemplate<int> int_params(&int_params_data[0], 2, 3);
    collection.AddBuffer< MatrixBufferTemplate<int> >(int_params_key, int_params);

    LinearMatrixFeature<double, int> matrix_feature(  float_params_key, int_params_key, 
                                                      indices_key, xs_key);
    matrix_feature.Bind(stack);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(0, 0), 0);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(0, 1), 5);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(0, 2), 10);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(1, 0), 3);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(1, 1), 8);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(1, 2), 13);
}

BOOST_AUTO_TEST_CASE(test_FeatureValue_linear_combination)
{
    double float_params_data[] = {0, 0, -1.0, 0.0, 1.0, 2.0, -3.0,
                                  0, 0, -0.5, 0.5, 1.5, 1, 1};
    MatrixBufferTemplate<double> float_params(&float_params_data[0], 2, 7);
    collection.AddBuffer< MatrixBufferTemplate<double> >(float_params_key, float_params);

    int int_params_data[] = {MATRIX_FEATURES, 5, 0, 1, 2, 3, 4,
                             MATRIX_FEATURES, 3, 0, 2, 4, 1, 1};
    MatrixBufferTemplate<int> int_params(&int_params_data[0], 2, 7);
    collection.AddBuffer< MatrixBufferTemplate<int> >(int_params_key, int_params);

    LinearMatrixFeature<double, int> matrix_feature(  float_params_key, int_params_key, 
                                                      indices_key, xs_key);
    matrix_feature.Bind(stack);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(0, 0), -4);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(0, 1), -9); 
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(1, 0), 7);
    BOOST_CHECK_EQUAL(matrix_feature.FeatureValue(1, 1), 14.5);
}

BOOST_AUTO_TEST_CASE(test_FeatureExtractor_linear_combination)
{
    double float_params_data[] = {0, 0, -1.0, 0.0, 1.0, 2.0, -3.0,
                                  0, 0, -0.5, 0.5, 1.5, 1, 1};
    MatrixBufferTemplate<double> float_params(&float_params_data[0], 2, 7);
    collection.AddBuffer< MatrixBufferTemplate<double> >(float_params_key, float_params);

    int int_params_data[] = {MATRIX_FEATURES, 5, 0, 1, 2, 3, 4,
                             MATRIX_FEATURES, 3, 0, 2, 4, 1, 1};
    MatrixBufferTemplate<int> int_params(&int_params_data[0], 2, 7);
    collection.AddBuffer< MatrixBufferTemplate<int> >(int_params_key, int_params);

    LinearMatrixFeature<double, int> matrix_feature(  float_params_key, int_params_key, 
                                                      indices_key, xs_key);

    FeatureExtractorStep< LinearMatrixFeature<double, int> > fe(matrix_feature, FEATURES_BY_DATAPOINTS);

    fe.ProcessStep(stack, collection);

    // MatrixBufferTemplate<double> feature_values = 
    //       collection.GetBuffer< MatrixBufferTemplate<double> >(fe.FeatureValuesBufferId);

    // BOOST_CHECK_EQUAL(feature_values.Get(0, 0), -4);
    // BOOST_CHECK_EQUAL(feature_values.Get(0, 1), -9); 
    // BOOST_CHECK_EQUAL(feature_values.Get(1, 0), 7);
    // BOOST_CHECK_EQUAL(feature_values.Get(1, 1), 14.5);
}


BOOST_AUTO_TEST_SUITE_END()
