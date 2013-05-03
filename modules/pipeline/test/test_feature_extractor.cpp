#include <boost/test/unit_test.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "FeatureExtractorStep.h"

template <class FloatType, class IntType>
class TestFeatureBinding
{
public:
    TestFeatureBinding()
    {}
    ~TestFeatureBinding()
    {}

    FloatType FeatureValue( const int featureIndex, const int relativeSampleIndex) const
    {
        return static_cast<FloatType>(featureIndex * GetNumberOfDatapoints() + relativeSampleIndex);
    }

    IntType GetNumberOfFeatures() const
    {
        return 3;
    }

    IntType GetNumberOfDatapoints() const
    {
        return 5;
    }
};


template <class FloatType, class IntType>
class TestFeature
{
public:
    TestFeature()
    {}
    ~TestFeature()
    {}

    TestFeatureBinding<FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const
    {
        UNUSED_PARAM(readCollection);
        TestFeatureBinding<FloatType, IntType> result;
        return result;
    }

    typedef FloatType Float;
    typedef IntType Int;
    typedef TestFeatureBinding<FloatType,IntType> FeatureBinding;
};

BOOST_AUTO_TEST_SUITE( FeatureExtractorTests )

BOOST_AUTO_TEST_CASE(test_ProcessStep_features_by_datapoints)
{
    BufferCollection collection;
    BufferCollectionStack stack;

    TestFeature<double,int> test_feature;
    FeatureExtractorStep< TestFeature<double,int> > feature_extractor(test_feature, FEATURES_BY_DATAPOINTS);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(feature_extractor.FeatureValuesBufferId));
    boost::mt19937 gen(0);
    feature_extractor.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(feature_extractor.FeatureValuesBufferId));

    MatrixBufferTemplate<double>& feature_values =
              collection.GetBuffer< MatrixBufferTemplate<double> >(feature_extractor.FeatureValuesBufferId);

    BOOST_CHECK_EQUAL(feature_values.GetM(), 3);
    BOOST_CHECK_EQUAL(feature_values.GetN(), 5);

    double data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    MatrixBufferTemplate<double> expect_feature_values(&data[0], 3, 5);

    BOOST_CHECK(feature_values == expect_feature_values);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_datapoints_by_features)
{
    BufferCollection collection;
    BufferCollectionStack stack;

    TestFeature<double,int> test_feature;
    FeatureExtractorStep< TestFeature<double,int> > feature_extractor(test_feature, DATAPOINTS_BY_FEATURES);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(feature_extractor.FeatureValuesBufferId));
    boost::mt19937 gen(0);
    feature_extractor.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(feature_extractor.FeatureValuesBufferId));

    MatrixBufferTemplate<double>& feature_values =
              collection.GetBuffer< MatrixBufferTemplate<double> >(feature_extractor.FeatureValuesBufferId);

    BOOST_CHECK_EQUAL(feature_values.GetM(), 5);
    BOOST_CHECK_EQUAL(feature_values.GetN(), 3);

    double data[] = {0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14};
    MatrixBufferTemplate<double> expect_feature_values(&data[0], 5, 3);

    double transpose_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    MatrixBufferTemplate<double> expect_transpose_feature_values(&transpose_data[0], 3, 5);

    BOOST_CHECK(feature_values == expect_feature_values);
    BOOST_CHECK(feature_values == expect_transpose_feature_values.Transpose());
}

BOOST_AUTO_TEST_SUITE_END()