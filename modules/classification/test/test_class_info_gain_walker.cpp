#include <boost/test/unit_test.hpp>

#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "ClassInfoGainWalker.h"
#include "BestSplitpointsWalkingSortedStep.h"


struct ClassInfoGainWalkerFixture {
    ClassInfoGainWalkerFixture()
    : fm_key("feature")
    , classes_key("classes")
    , weights_key("weights")
    , number_of_classes(3)
    , collection()
    , stack()
    {
        float fm_data[] = {0.06, 0.03, 0.07, 0.01, 0.02, -0.15, -0.25, 0,
                            8.9,  3000,    2, 6.1,    2,    -9,     2,  2};
        const MatrixBufferTemplate<float> fm = MatrixBufferTemplate<float>(&fm_data[0], 2, 8);
        collection.AddBuffer(fm_key, fm);

        int classes_data[] = {0,1,2,1,1,2,0,2};
        const VectorBufferTemplate<int> classes = VectorBufferTemplate<int>(&classes_data[0], 8);
        collection.AddBuffer(classes_key, classes);

        float weights_data[] = {1,1,1,1,1,1,1,1};
        const VectorBufferTemplate<float> weights = VectorBufferTemplate<float>(&weights_data[0], 8);
        collection.AddBuffer(weights_key, weights);

        stack.Push(&collection);
    }

    ~ClassInfoGainWalkerFixture()
    {
    }

    const BufferCollectionKey_t fm_key;
    const BufferCollectionKey_t classes_key;
    const BufferCollectionKey_t weights_key;
    const int number_of_classes;
    BufferCollection collection;
    BufferCollectionStack stack;

typedef BufferTypes<float, int, int, float, int, float, float, int, float> BufferTypes_t;
};

BOOST_FIXTURE_TEST_SUITE( ClassInfoGainWalkerTests,  ClassInfoGainWalkerFixture )

BOOST_AUTO_TEST_CASE(test_ClassInfoGainWalker_Impurity)
{
    ClassInfoGainWalker< BufferTypes_t > classInfoGainWalker(weights_key, classes_key, number_of_classes);
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
    ClassInfoGainWalker< BufferTypes_t > classInfoGainWalker(weights_key, classes_key, number_of_classes);
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
    ClassInfoGainWalker< BufferTypes_t > classInfoGainWalker(weights_key, classes_key, number_of_classes);
    BestSplitpointsWalkingSortedStep< ClassInfoGainWalker<BufferTypes_t> > bestsplits(classInfoGainWalker, fm_key, FEATURES_BY_DATAPOINTS);
    boost::mt19937 gen(0);
    bestsplits.ProcessStep(stack, collection, gen);

    BOOST_CHECK( collection.HasBuffer< MatrixBufferTemplate<float> >( bestsplits.SplitpointBufferId ) );
    MatrixBufferTemplate<float>& best_thresholds = collection.GetBuffer< MatrixBufferTemplate<float> >( bestsplits.SplitpointBufferId );
    float expected_best_thresholds_data[] = {0.005, 4.05};
    MatrixBufferTemplate<float> expected_best_thresholds = MatrixBufferTemplate<float>(&expected_best_thresholds_data[0], 2, 1);
    BOOST_CHECK(best_thresholds == expected_best_thresholds);

    BOOST_CHECK( collection.HasBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId ) );
    VectorBufferTemplate<int>& threshold_counts = collection.GetBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId );
    int expected_threshold_counts_data[] = {1, 1};
    VectorBufferTemplate<int> expected_threshold_counts = VectorBufferTemplate<int>(&expected_threshold_counts_data[0], 2);
    BOOST_CHECK(threshold_counts == expected_threshold_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<float> >( bestsplits.ChildCountsBufferId ) );
    Tensor3BufferTemplate<float>& child_counts = collection.GetBuffer< Tensor3BufferTemplate<float> >( bestsplits.ChildCountsBufferId );
    float expected_child_counts_data[] = {5, 3,
                                          3, 5};
    Tensor3BufferTemplate<float> expected_child_counts = Tensor3BufferTemplate<float>(&expected_child_counts_data[0], 2, 1, 2);
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