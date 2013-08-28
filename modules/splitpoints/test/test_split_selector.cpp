#include <boost/test/unit_test.hpp>

#include <vector>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "ShouldSplitNoCriteria.h"
#include "MinChildSizeCriteria.h"
#include "ClassEstimatorFinalizer.h"
#include "SplitSelectorBuffers.h"
#include "SplitBuffersIndices.h"
#include "SplitSelectorInfo.h"
#include "SplitSelector.h"

template<typename T>
VectorBufferTemplate<T> CreateVector(T data[], int n)
{
    return VectorBufferTemplate<T>(&data[0], n);
}

template<typename T>
MatrixBufferTemplate<T> CreateMatrix(T data[], int m, int n)
{
    return MatrixBufferTemplate<T>(&data[0], m, n);
}

template<typename T>
Tensor3BufferTemplate<T> CreateTensor3(T data[], int l, int m, int n)
{
    return Tensor3BufferTemplate<T>(&data[0], l, m, n);
}

template<typename T>
MatrixBufferTemplate<T> CreateImpurityMatrix3x4()
{
    T data[] = {0.01, 1111, 1111, 1111,
                55, 6, -8, 2,
                20, 10, 30, 1111};
    return MatrixBufferTemplate<T>(&data[0], 3, 4);
}

template<typename T>
MatrixBufferTemplate<T> CreateImpurityMatrix3x4_2()
{
    T data[] = {-0.01, 1111, 1111, 1111,
                55, 6, -8, 220,
                20, 10, 30, 1111};
    return MatrixBufferTemplate<T>(&data[0], 3, 4);
}

template<typename T>
VectorBufferTemplate<T> CreateNumberOfSplitpointsVector3()
{
    T data[] = {1,4,3};
    return VectorBufferTemplate<T>(&data[0], 3);
}

template<typename T>
MatrixBufferTemplate<T> CreateSplitpointsMatrix3x4()
{
    T data[] = {1.5, 1111, 1111, 1111,
                -8, 3, 12, 0.5,
                13, -1.5, 2, 1111};
    return MatrixBufferTemplate<T>(&data[0], 3, 4);
}

template<typename T>
Tensor3BufferTemplate<T> CreateChildCountsTensor3x4x2()
{
    T data[] = {11,8, 0,0, 0,0, 0,0,
                1,6, 22,21, 9,5, 2,4,
                2,1, 5,4, 11,12, 0,0};
    return Tensor3BufferTemplate<T>(&data[0], 3, 4, 2);
}

template<typename T>
Tensor3BufferTemplate<T> CreateLeftTensor3x4x3()
{
    T data[] = {2,3,5, 0,0,0, 0,0,0, 0,0,0,
                2,10,8, 3,3,4, 2,7,1, 4,2,4,
                6,3,1, 8,8,4, 5,0,5, 0,0,0 };

    return Tensor3BufferTemplate<T>(&data[0], 3, 4, 3);
}

template<typename T>
Tensor3BufferTemplate<T> CreateRightTensor3x4x3()
{
    T data[] = {2,3,5, 0,0,0, 0,0,0, 0,0,0,
                2,10,8, 3,3,4, 2,7,1, 4,2,4,
                6,3,1, 8,8,4, 5,0,5, 0,0,0 };

    return Tensor3BufferTemplate<T>(&data[0], 3, 4, 3);
}

template<typename T>
MatrixBufferTemplate<T> CreateFeatureIntParamsMatrix3x2()
{
    T data[] = {1,5,
                2,7,
                8,1 };
    return MatrixBufferTemplate<T>(&data[0], 3, 2);
}

template<typename T>
MatrixBufferTemplate<T> CreateFeatureFloatParamsMatrix3x2()
{
    T data[] = {0.5, 2.5,
                1.5, 4.4,
                3.2, 1.2 };
    return MatrixBufferTemplate<T>(&data[0], 3, 2);
}

template<typename T>
MatrixBufferTemplate<T> CreateFeatureValues3x5()
{
    T data[] = {1.5, -1, 28, 2, 0,
                -8, 3, 12, 0.5, 0.9,
                13, -1.5, 2, 1.9, 2.1 };
    return MatrixBufferTemplate<T>(&data[0], 3, 5);
}

template<typename T>
VectorBufferTemplate<T> CreateIndices5()
{
    T data[] = {0, 1, 2, 3, 4};
    return VectorBufferTemplate<T>(&data[0], 5);
}

struct SplitSelectorFixture {
    SplitSelectorFixture()
    : im_key("im")
    , im_2_key("im2")
    , number_splitpoints_key("number_splitpoints")
    , splitpoints_key("splitpoints")
    , childcounts_key("childcounts")
    , left_key("left")
    , right_key("right")
    , feature_floatparams_key("feature_floatparams")
    , feature_intparams_key("feature_intparams")
    , feature_values_key("feature_values_key")
    , indices_key("indices")
    , im( CreateImpurityMatrix3x4<double>() )
    , im_2( CreateImpurityMatrix3x4_2<double>() )
    , number_splitpoints( CreateNumberOfSplitpointsVector3<int>() )
    , splitpoints( CreateSplitpointsMatrix3x4<float>() )
    , childcounts( CreateChildCountsTensor3x4x2<float>() )
    , left( CreateLeftTensor3x4x3<double>() )
    , right( CreateRightTensor3x4x3<double>() )
    , feature_floatparams( CreateFeatureFloatParamsMatrix3x2<double>() )
    , feature_intparams( CreateFeatureIntParamsMatrix3x2<int>() )
    , feature_values( CreateFeatureValues3x5<float>() )
    , indices( CreateIndices5<int>() )
    , collection()
    , stack()
    {
        collection.AddBuffer(im_key, im);
        collection.AddBuffer(im_2_key, im_2);
        collection.AddBuffer(number_splitpoints_key, number_splitpoints);
        collection.AddBuffer(splitpoints_key, splitpoints);
        collection.AddBuffer(childcounts_key, childcounts);
        collection.AddBuffer(left_key, left);
        collection.AddBuffer(right_key, right);
        collection.AddBuffer(feature_floatparams_key, feature_floatparams);
        collection.AddBuffer(feature_intparams_key, feature_intparams);
        collection.AddBuffer(feature_values_key, feature_values);
        collection.AddBuffer(indices_key, indices);
        stack.Push(&collection);
    }

    ~SplitSelectorFixture()
    {
    }

    typedef BufferTypes<float, int, int, double, int, float, double, int, double> BufferTypes_t;

    const BufferCollectionKey_t im_key;
    const BufferCollectionKey_t im_2_key;
    const BufferCollectionKey_t number_splitpoints_key;
    const BufferCollectionKey_t splitpoints_key;
    const BufferCollectionKey_t childcounts_key;
    const BufferCollectionKey_t left_key;
    const BufferCollectionKey_t right_key;
    const BufferCollectionKey_t feature_floatparams_key;
    const BufferCollectionKey_t feature_intparams_key;
    const BufferCollectionKey_t feature_values_key;
    const BufferCollectionKey_t indices_key;
    const MatrixBufferTemplate<double> im;
    const MatrixBufferTemplate<double> im_2;
    const VectorBufferTemplate<int> number_splitpoints;
    const MatrixBufferTemplate<float> splitpoints;
    const Tensor3BufferTemplate<float> childcounts;
    const Tensor3BufferTemplate<double> left;
    const Tensor3BufferTemplate<double> right;
    const MatrixBufferTemplate<double> feature_floatparams;
    const MatrixBufferTemplate<int> feature_intparams;
    const MatrixBufferTemplate<float> feature_values;
    const VectorBufferTemplate<int> indices;
    BufferCollection collection;
    BufferCollectionStack stack;
};


BOOST_FIXTURE_TEST_SUITE( SplitSelectorTests, SplitSelectorFixture )

BOOST_AUTO_TEST_CASE(test_ProcessSplit_NoCriteria)
{
    SplitSelectorBuffers buffers(im_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              NullKey, FEATURES_BY_DATAPOINTS, NULL);
    std::vector<SplitSelectorBuffers> split_select_buffers;
    split_select_buffers.push_back(buffers);

    ShouldSplitNoCriteria no_criteria;
    ClassEstimatorFinalizer<BufferTypes_t> classEsimatorFinalizer;
    SplitSelector<BufferTypes_t> splitselector(split_select_buffers, &no_criteria, &classEsimatorFinalizer, NULL);


    const int depth = 5;
    BufferCollection bc;
    SplitSelectorInfo<BufferTypes_t> selectorInfo = splitselector.ProcessSplits(stack, depth, bc, 0);
    BOOST_CHECK( selectorInfo.ValidSplit() );

    const int nodeId = 0;
    const int leftNodeId = 2;
    const int rightNodeId = 5;
    VectorBufferTemplate<float> counts(10);
    VectorBufferTemplate<int> depths(10);
    MatrixBufferTemplate<double> floatParams(10,2);
    MatrixBufferTemplate<int> intParams(10,2);
    MatrixBufferTemplate<float> estimatorParams(10,3);
    selectorInfo.WriteToTree(nodeId, leftNodeId, rightNodeId, counts, depths, floatParams, intParams, estimatorParams);

    const int bestFeature = 1;
    const int bestSplitpoint = 0;

    BOOST_CHECK_EQUAL( counts.Get(leftNodeId), childcounts.Get(bestFeature, bestSplitpoint, LEFT_CHILD_INDEX) );
    BOOST_CHECK_EQUAL( counts.Get(rightNodeId), childcounts.Get(bestFeature, bestSplitpoint, RIGHT_CHILD_INDEX) );
    BOOST_CHECK_EQUAL( depths.Get(nodeId), depth );
    BOOST_CHECK_EQUAL( depths.Get(leftNodeId), depth+1 );
    BOOST_CHECK_EQUAL( depths.Get(rightNodeId), depth+1 );
    BOOST_CHECK_EQUAL( floatParams.Get(nodeId, SPLITPOINT_INDEX), splitpoints.Get(bestFeature, bestSplitpoint));
    BOOST_CHECK_EQUAL( floatParams.Get(nodeId, 1), feature_floatparams.Get(bestFeature, 1));
    BOOST_CHECK( intParams.SliceRowAsVector(nodeId) == feature_intparams.SliceRowAsVector(bestFeature));

    const VectorBufferTemplate<float> left_expected = ConvertVectorBufferTemplate<double, float>(left.SliceRow(bestFeature, bestSplitpoint).Normalized());
    const VectorBufferTemplate<float> right_expected = ConvertVectorBufferTemplate<double, float>(right.SliceRow(bestFeature, bestSplitpoint).Normalized());

    BOOST_CHECK( estimatorParams.SliceRowAsVector(leftNodeId) == left_expected);
    BOOST_CHECK( estimatorParams.SliceRowAsVector(rightNodeId) == right_expected);
}

BOOST_AUTO_TEST_CASE(test_ProcessSplit_NoValidSplit)
{
    SplitSelectorBuffers buffers(im_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              NullKey, FEATURES_BY_DATAPOINTS, NULL);
    std::vector<SplitSelectorBuffers> split_select_buffers;
    split_select_buffers.push_back(buffers);

    MinChildSizeCriteria min_child_size_criteria(1000);
    ClassEstimatorFinalizer<BufferTypes_t> classEsimatorFinalizer;
    SplitSelector<BufferTypes_t> splitselector(split_select_buffers, &min_child_size_criteria, &classEsimatorFinalizer);

    const int depth = 5;
    BufferCollection bc;
    SplitSelectorInfo<BufferTypes_t> selectorInfo = splitselector.ProcessSplits(stack, depth, bc, 0);
    BOOST_CHECK( !selectorInfo.ValidSplit() );
}

BOOST_AUTO_TEST_CASE(test_ProcessSplit_MinChildSizeCriteria)
{
    SplitSelectorBuffers buffers(im_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              NullKey, FEATURES_BY_DATAPOINTS, NULL);
    std::vector<SplitSelectorBuffers> split_select_buffers;
    split_select_buffers.push_back(buffers);

    MinChildSizeCriteria min_child_size_criteria(10);
    ClassEstimatorFinalizer<BufferTypes_t> classEsimatorFinalizer;
    SplitSelector<BufferTypes_t> splitselector(split_select_buffers, &min_child_size_criteria, &classEsimatorFinalizer);

    const int depth = 5;
    BufferCollection bc;
    SplitSelectorInfo<BufferTypes_t> selectorInfo = splitselector.ProcessSplits(stack, depth, bc, 0);
    BOOST_CHECK( selectorInfo.ValidSplit() );

    const int nodeId = 0;
    const int leftNodeId = 4;
    const int rightNodeId = 3;
    VectorBufferTemplate<float> counts(10);
    VectorBufferTemplate<int> depths(10);
    MatrixBufferTemplate<double> floatParams(10,2);
    MatrixBufferTemplate<int> intParams(10,2);
    MatrixBufferTemplate<float> estimatorParams(10,3);
    selectorInfo.WriteToTree(nodeId, leftNodeId, rightNodeId, counts, depths, floatParams, intParams, estimatorParams);

    const int bestFeature = 2;
    const int bestSplitpoint = 2;

    BOOST_CHECK_EQUAL( counts.Get(leftNodeId), childcounts.Get(bestFeature, bestSplitpoint, LEFT_CHILD_INDEX) );
    BOOST_CHECK_EQUAL( counts.Get(rightNodeId), childcounts.Get(bestFeature, bestSplitpoint, RIGHT_CHILD_INDEX) );
    BOOST_CHECK_EQUAL( depths.Get(nodeId), depth );
    BOOST_CHECK_EQUAL( depths.Get(leftNodeId), depth+1 );
    BOOST_CHECK_EQUAL( depths.Get(rightNodeId), depth+1 );
    BOOST_CHECK_EQUAL( floatParams.Get(nodeId, SPLITPOINT_INDEX), splitpoints.Get(bestFeature, bestSplitpoint));
    BOOST_CHECK_EQUAL( floatParams.Get(nodeId, 1), feature_floatparams.Get(bestFeature, 1));
    BOOST_CHECK( intParams.SliceRowAsVector(nodeId) == feature_intparams.SliceRowAsVector(bestFeature));

    const VectorBufferTemplate<float> left_expected = ConvertVectorBufferTemplate<double, float>(left.SliceRow(bestFeature, bestSplitpoint).Normalized());
    const VectorBufferTemplate<float> right_expected = ConvertVectorBufferTemplate<double, float>(right.SliceRow(bestFeature, bestSplitpoint).Normalized());

    BOOST_CHECK( estimatorParams.SliceRowAsVector(leftNodeId) == left_expected);
    BOOST_CHECK( estimatorParams.SliceRowAsVector(rightNodeId) == right_expected);
}

BOOST_AUTO_TEST_CASE(test_ProcessSplit_two_SplitSelectorBuffers)
{
    SplitSelectorBuffers buffers(im_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              NullKey, FEATURES_BY_DATAPOINTS, NULL);
    SplitSelectorBuffers buffers2(im_2_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              NullKey, FEATURES_BY_DATAPOINTS, NULL);
    std::vector<SplitSelectorBuffers> split_select_buffers;
    split_select_buffers.push_back(buffers);
    split_select_buffers.push_back(buffers2);

    ShouldSplitNoCriteria no_criteria;
    ClassEstimatorFinalizer<BufferTypes_t> classEsimatorFinalizer;
    SplitSelector<BufferTypes_t> splitselector(split_select_buffers, &no_criteria, &classEsimatorFinalizer);


    const int depth = 5;
    BufferCollection bc;
    SplitSelectorInfo<BufferTypes_t> selectorInfo = splitselector.ProcessSplits(stack, depth, bc, 0);
    BOOST_CHECK( selectorInfo.ValidSplit() );

    const int nodeId = 0;
    const int leftNodeId = 2;
    const int rightNodeId = 5;
    VectorBufferTemplate<float> counts(10);
    VectorBufferTemplate<int> depths(10);
    MatrixBufferTemplate<double> floatParams(10,2);
    MatrixBufferTemplate<int> intParams(10,2);
    MatrixBufferTemplate<float> estimatorParams(10,3);
    selectorInfo.WriteToTree(nodeId, leftNodeId, rightNodeId, counts, depths, floatParams, intParams, estimatorParams);

    const int bestFeature = 1;
    const int bestSplitpoint = 3;

    BOOST_CHECK_EQUAL( counts.Get(leftNodeId), childcounts.Get(bestFeature, bestSplitpoint, LEFT_CHILD_INDEX) );
    BOOST_CHECK_EQUAL( counts.Get(rightNodeId), childcounts.Get(bestFeature, bestSplitpoint, RIGHT_CHILD_INDEX) );
    BOOST_CHECK_EQUAL( depths.Get(nodeId), depth );
    BOOST_CHECK_EQUAL( depths.Get(leftNodeId), depth+1 );
    BOOST_CHECK_EQUAL( depths.Get(rightNodeId), depth+1 );
    BOOST_CHECK_EQUAL( floatParams.Get(nodeId, SPLITPOINT_INDEX), splitpoints.Get(bestFeature, bestSplitpoint));
    BOOST_CHECK_EQUAL( floatParams.Get(nodeId, 1), feature_floatparams.Get(bestFeature, 1));
    BOOST_CHECK( intParams.SliceRowAsVector(nodeId) == feature_intparams.SliceRowAsVector(bestFeature));


    const VectorBufferTemplate<float> left_expected = ConvertVectorBufferTemplate<double, float>(left.SliceRow(bestFeature, bestSplitpoint).Normalized());
    const VectorBufferTemplate<float> right_expected = ConvertVectorBufferTemplate<double, float>(right.SliceRow(bestFeature, bestSplitpoint).Normalized());

    BOOST_CHECK( estimatorParams.SliceRowAsVector(leftNodeId) == left_expected);
    BOOST_CHECK( estimatorParams.SliceRowAsVector(rightNodeId) == right_expected);
}

BOOST_AUTO_TEST_CASE(test_SplitBuffers_FEATURES_BY_DATAPOINTS)
{
    SplitSelectorBuffers buffers(im_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              feature_values_key, FEATURES_BY_DATAPOINTS, NULL);
    std::vector<SplitSelectorBuffers> split_select_buffers;
    split_select_buffers.push_back(buffers);

    MinChildSizeCriteria min_child_size_criteria(10);
    ClassEstimatorFinalizer<BufferTypes_t> classEsimatorFinalizer;
    SplitBuffersIndices<BufferTypes_t> splitIndices(indices_key);
    SplitSelector<BufferTypes_t> splitselector(split_select_buffers, &min_child_size_criteria, &classEsimatorFinalizer, &splitIndices);

    const int depth = 5;
    BufferCollection bc;
    SplitSelectorInfo<BufferTypes_t> selectorInfo = splitselector.ProcessSplits(stack, depth, bc, 0);
    BOOST_CHECK( selectorInfo.ValidSplit() );

    BufferCollection leftBufCol;
    BufferCollection rightBufCol;
    float leftSize, rightSize;
    selectorInfo.SplitBuffers(leftBufCol, rightBufCol, leftSize, rightSize);
    BOOST_CHECK_CLOSE(leftSize, 11.0, 0.1);
    BOOST_CHECK_CLOSE(rightSize, 12.0, 0.1);

    int leftExpectedIndexData[] = {0, 4};
    int rightExpectedIndexData[] = {1, 2, 3};

    BOOST_CHECK(leftBufCol.GetBuffer< VectorBufferTemplate<int> >(indices_key) == CreateVector<int>(leftExpectedIndexData, 2));
    BOOST_CHECK(rightBufCol.GetBuffer< VectorBufferTemplate<int> >(indices_key) == CreateVector<int>(rightExpectedIndexData, 3));
}

BOOST_AUTO_TEST_CASE(test_SplitBuffers_DATAPOINTS_BY_FEATURES)
{
    MatrixBufferTemplate<float>& fv = collection.GetBuffer< MatrixBufferTemplate<float> >(feature_values_key);
    fv = fv.Transpose();

    SplitSelectorBuffers buffers(im_key, splitpoints_key, number_splitpoints_key, childcounts_key,
                              left_key, right_key, feature_floatparams_key, feature_intparams_key,
                              feature_values_key, DATAPOINTS_BY_FEATURES, NULL);
    std::vector<SplitSelectorBuffers> split_select_buffers;
    split_select_buffers.push_back(buffers);

    MinChildSizeCriteria min_child_size_criteria(10);
    ClassEstimatorFinalizer<BufferTypes_t> classEsimatorFinalizer;
    SplitBuffersIndices<BufferTypes_t> splitIndices(indices_key);
    SplitSelector<BufferTypes_t> splitselector(split_select_buffers, &min_child_size_criteria, &classEsimatorFinalizer, &splitIndices);

    const int depth = 5;
    BufferCollection bc;
    SplitSelectorInfo<BufferTypes_t> selectorInfo = splitselector.ProcessSplits(stack, depth, bc, 0);
    BOOST_CHECK( selectorInfo.ValidSplit() );

    BufferCollection leftBufCol;
    BufferCollection rightBufCol;
    float leftSize, rightSize;
    selectorInfo.SplitBuffers(leftBufCol, rightBufCol, leftSize, rightSize);
    BOOST_CHECK_CLOSE(leftSize, 11.0, 0.1);
    BOOST_CHECK_CLOSE(rightSize, 12.0, 0.1);

    int leftExpectedIndexData[] = {0, 4};
    int rightExpectedIndexData[] = {1, 2, 3};

    BOOST_CHECK(leftBufCol.GetBuffer< VectorBufferTemplate<int> >(indices_key) == CreateVector<int>(leftExpectedIndexData, 2));
    BOOST_CHECK(rightBufCol.GetBuffer< VectorBufferTemplate<int> >(indices_key) == CreateVector<int>(rightExpectedIndexData, 3));
}


BOOST_AUTO_TEST_SUITE_END()