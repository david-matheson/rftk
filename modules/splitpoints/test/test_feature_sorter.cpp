#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "FeatureSorter.h"

BOOST_AUTO_TEST_SUITE( FeatureSorterTests )

template<typename T>
MatrixBufferTemplate<T> CreateExampleMatrix()
{
    T data[] = {6, 3, 7, 0, 2,
                6, 3, 6, 8.9, 2, 
                9, 3, 6, 8.8, 8.75};

    return MatrixBufferTemplate<T>(&data[0], 3, 5);
}

BOOST_AUTO_TEST_CASE(test_Feature_Sorter_GetUnSortedIndex_FEATURES_BY_DATAPOINTS)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();

    FeatureSorter<double> fs(mb, FEATURES_BY_DATAPOINTS, 0);
    fs.Sort();
    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(0), 3);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(0), 0);

    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(1), 4);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(1), 2);

    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(2), 1);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(2), 3);

    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(3), 0);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(3), 6);

    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(4), 2);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(4), 7);

    FeatureSorter<double> fs2(mb, FEATURES_BY_DATAPOINTS, 1);
    fs2.Sort();
    BOOST_CHECK_EQUAL(fs2.GetUnSortedIndex(0), 4);
    BOOST_CHECK_EQUAL(fs2.GetFeatureValue(0), 2);

    BOOST_CHECK_EQUAL(fs2.GetUnSortedIndex(1), 1);
    BOOST_CHECK_EQUAL(fs2.GetFeatureValue(1), 3);

    BOOST_CHECK_EQUAL(fs2.GetUnSortedIndex(2), 0);
    BOOST_CHECK_EQUAL(fs2.GetFeatureValue(2), 6);

    BOOST_CHECK_EQUAL(fs2.GetUnSortedIndex(3), 2);
    BOOST_CHECK_EQUAL(fs2.GetFeatureValue(3), 6);

    BOOST_CHECK_EQUAL(fs2.GetUnSortedIndex(4), 3);
    BOOST_CHECK_EQUAL(fs2.GetFeatureValue(4), 8.9);
}

BOOST_AUTO_TEST_CASE(test_Feature_Sorter_GetUnSortedIndex_DATAPOINTS_BY_FEATURES)
{
    MatrixBufferTemplate<double> mb = CreateExampleMatrix<double>();

    FeatureSorter<double> fs(mb, DATAPOINTS_BY_FEATURES, 3);
    fs.Sort();
    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(0), 0);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(0), 0);

    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(1), 2);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(1), 8.8);

    BOOST_CHECK_EQUAL(fs.GetUnSortedIndex(2), 1);
    BOOST_CHECK_EQUAL(fs.GetFeatureValue(2), 8.9);
}

BOOST_AUTO_TEST_SUITE_END()