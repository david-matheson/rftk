#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "ClassInfoGainWalker.h"
#include "BestSplitpointsWalkingSortedStep.h"

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
MatrixBufferTemplate<T> CreateFeatureMatrix()
{
    T data[] = {0.06, 0.03, 0.07, 0.01, 0.02, -0.15, -0.25, 0,
                8.9,  3000,    2, 6.1,    2,    -9,     2,  2};

    return MatrixBufferTemplate<T>(&data[0], 2, 8);
}

template<typename T>
VectorBufferTemplate<T> CreateClassesVector()
{
    T data[] = {0,1,2,1,1,2,0,2};

    return VectorBufferTemplate<T>(&data[0], 8);
}

template<typename T>
VectorBufferTemplate<T> CreateWeightsVector()
{
    T data[] = {1,1,1,1,1,1,1,1};

    return VectorBufferTemplate<T>(&data[0], 8);
}

struct ClassInfoGainWalkerFixture {
    ClassInfoGainWalkerFixture()
    : fm_key("feature")
    , fm( CreateFeatureMatrix<float>() )
    , classes_key("classes")
    , classes( CreateClassesVector<int>() )
    , weights_key("weights")
    , weights( CreateWeightsVector<float>() )
    , number_of_classes(3)
    , collection()
    , stack()
    {
        collection.AddBuffer(fm_key, fm);
        collection.AddBuffer(classes_key, classes);
        collection.AddBuffer(weights_key, weights);
        stack.Push(&collection);
    }

    ~ClassInfoGainWalkerFixture()
    {
    }

    const BufferCollectionKey_t fm_key;
    const MatrixBufferTemplate<float> fm;
    const BufferCollectionKey_t classes_key;
    const VectorBufferTemplate<int> classes;
    const BufferCollectionKey_t weights_key;
    const VectorBufferTemplate<float> weights;
    const int number_of_classes;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( ClassInfoGainWalkerTests,  ClassInfoGainWalkerFixture )

BOOST_AUTO_TEST_CASE(test_ClassInfoGainWalker_Impurity)
{
    ClassInfoGainWalker<float, int> classInfoGainWalker(weights_key, classes_key, number_of_classes);
    classInfoGainWalker.Bind(stack);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.0, 1);

    classInfoGainWalker.MoveLeftToRight(2);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.199, 1);

    classInfoGainWalker.MoveLeftToRight(5);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.467, 1);

    classInfoGainWalker.MoveLeftToRight(7);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.949, 1);

    classInfoGainWalker.MoveLeftToRight(3);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.656, 1);

    classInfoGainWalker.MoveLeftToRight(6);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.360, 1);

    classInfoGainWalker.MoveLeftToRight(0);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.4669, 1);

    classInfoGainWalker.MoveLeftToRight(1);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.199, 1);

    classInfoGainWalker.MoveLeftToRight(4);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.0, 1);
}

BOOST_AUTO_TEST_CASE(test_ClassInfoGainWalker_Reset)
{
    ClassInfoGainWalker<float, int> classInfoGainWalker(weights_key, classes_key, number_of_classes);
    classInfoGainWalker.Bind(stack);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.0, 1);

    classInfoGainWalker.MoveLeftToRight(2);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.199, 1);

    classInfoGainWalker.Reset();
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.0, 1);

    classInfoGainWalker.MoveLeftToRight(2);
    BOOST_CHECK_CLOSE(classInfoGainWalker.Impurity(), 0.199, 1);
}

BOOST_AUTO_TEST_CASE(test_ClassInfoGainWalker_BestSplitpointsWalkingSortedStep_ProcessStep_FEATURES_BY_DATAPOINTS)
{
    ClassInfoGainWalker<float, int> classInfoGainWalker(weights_key, classes_key, number_of_classes);
    BestSplitpointsWalkingSortedStep< ClassInfoGainWalker<float, int> > bestsplits(classInfoGainWalker, fm_key, FEATURES_BY_DATAPOINTS);
    bestsplits.ProcessStep(stack, collection);

    BOOST_CHECK( collection.HasBuffer< MatrixBufferTemplate<float> >( bestsplits.SplitpointBufferId ) );
    MatrixBufferTemplate<float>& best_thresholds = collection.GetBuffer< MatrixBufferTemplate<float> >( bestsplits.SplitpointBufferId );
    float expected_best_thresholds_data[] = {0.005, 4.05};
    MatrixBufferTemplate<float> expected_best_thresholds = CreateMatrix<float>(expected_best_thresholds_data, 2, 1);
    BOOST_CHECK(best_thresholds == expected_best_thresholds);

    BOOST_CHECK( collection.HasBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId ) );
    VectorBufferTemplate<int>& threshold_counts = collection.GetBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId );
    int expected_threshold_counts_data[] = {1, 1};
    VectorBufferTemplate<int> expected_threshold_counts = CreateVector<int>(expected_threshold_counts_data, 2);
    BOOST_CHECK(threshold_counts == expected_threshold_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<float> >( bestsplits.ChildCountsBufferId ) );
    Tensor3BufferTemplate<float>& child_counts = collection.GetBuffer< Tensor3BufferTemplate<float> >( bestsplits.ChildCountsBufferId );
    float expected_child_counts_data[] = {5, 3,
                                          3, 5};
    Tensor3BufferTemplate<float> expected_child_counts = CreateTensor3<float>(expected_child_counts_data, 2, 1, 2);
    BOOST_CHECK(child_counts == expected_child_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<float> >( bestsplits.LeftYsBufferId ) );
    Tensor3BufferTemplate<float>& left_ys = collection.GetBuffer< Tensor3BufferTemplate<float> >( bestsplits.LeftYsBufferId );
    BOOST_CHECK_CLOSE(left_ys.Get(0,0,0), 0.2, 0.001);
    BOOST_CHECK_CLOSE(left_ys.Get(0,0,1), 0.6, 0.001);
    BOOST_CHECK_CLOSE(left_ys.Get(0,0,2), 0.2, 0.001);
    BOOST_CHECK_CLOSE(left_ys.Get(1,0,0), 0.333333, 0.001);
    BOOST_CHECK_CLOSE(left_ys.Get(1,0,1), 0.666667, 0.001);
    BOOST_CHECK_CLOSE(left_ys.Get(1,0,2), 0.0, 0.001);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<float> >( bestsplits.RightYsBufferId ) );
    Tensor3BufferTemplate<float>& right_ys = collection.GetBuffer< Tensor3BufferTemplate<float> >( bestsplits.RightYsBufferId );
    BOOST_CHECK_CLOSE(right_ys.Get(0,0,0), 0.333333, 0.001);
    BOOST_CHECK_CLOSE(right_ys.Get(0,0,1), 0.0, 0.001);
    BOOST_CHECK_CLOSE(right_ys.Get(0,0,2), 0.666667, 0.001);
    BOOST_CHECK_CLOSE(right_ys.Get(1,0,0), 0.2, 0.001);
    BOOST_CHECK_CLOSE(right_ys.Get(1,0,1), 0.2, 0.001);
    BOOST_CHECK_CLOSE(right_ys.Get(1,0,2), 0.6, 0.001);
}

BOOST_AUTO_TEST_SUITE_END()