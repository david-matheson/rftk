#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferTypes.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"
#include "SplitpointStatsStep.h"
#include "ClassStatsUpdater.h"

BOOST_AUTO_TEST_SUITE(SplitpointStatsStepTest)

BOOST_AUTO_TEST_CASE(test_ProcessStep_FEATURES_BY_DATAPOINTS)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId weights_key = "weights";
    float weight_data[] = {1.0, 0.5, 2.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    VectorBufferTemplate<float> weights(&weight_data[0], 10);
    bc.AddBuffer(weights_key, weights);

    BufferId classes_key = "classes";
    int classes_data[] = {0, 2, 1, 2, 1, 0, 0, 1, 1, 2};
    VectorBufferTemplate<int> classes(&classes_data[0], 10);
    bc.AddBuffer(classes_key, classes);

    const int numberOfClasses = 3;
    ClassStatsUpdater<SinglePrecisionBufferTypes> classStatsUpdater(weights_key, classes_key, numberOfClasses);

    BufferId split_points_key = "split_points";
    float split_points_data[] = {2.0, 5.5, 7.0,
                                   0.5, 1.5, 0,
                                   1.003, 0, 0 };
    MatrixBufferTemplate<float> split_points(&split_points_data[0], 3, 3);
    bc.AddBuffer(split_points_key, split_points);

    BufferId split_points_counts_key = "split_points_counts";
    int split_points_counts_data[] = {3,2,1};
    VectorBufferTemplate<int> split_points_counts(&split_points_counts_data[0], 3);
    bc.AddBuffer(split_points_counts_key, split_points_counts);

    BufferId feature_values_key = "feature_values";
    float feature_values_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                   1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 0, 0, 0, 0 };
    MatrixBufferTemplate<float> feature_values(&feature_values_data[0], 3, 10);
    bc.AddBuffer(feature_values_key, feature_values);

    SplitpointStatsStep< ClassStatsUpdater<SinglePrecisionBufferTypes> > splitpointStatsStep(split_points_key, 
                                                                              split_points_counts_key, 
                                                                              feature_values_key, 
                                                                              FEATURES_BY_DATAPOINTS,
                                                                              classStatsUpdater);
    boost::mt19937 gen;
    splitpointStatsStep.ProcessStep(stack, bc, gen, bc, 0);

    Tensor3BufferTemplate<float>& childCounts = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.ChildCountsBufferId);

    Tensor3BufferTemplate<float>& leftStats = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.LeftStatsBufferId);

    Tensor3BufferTemplate<float>& rightStats = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.RightStatsBufferId);

    BOOST_CHECK_CLOSE(childCounts.Get(0,0,LEFT_CHILD), 10, 0.1);
    BOOST_CHECK_CLOSE(childCounts.Get(0,0,RIGHT_CHILD), 1.5, 0.1);
    BOOST_CHECK_CLOSE(childCounts.Get(0,2,LEFT_CHILD), 3, 0.1);
    BOOST_CHECK_CLOSE(childCounts.Get(0,2,RIGHT_CHILD), 8.5, 0.1);
    BOOST_CHECK_CLOSE(childCounts.Get(2,0,LEFT_CHILD), 4, 0.1);
    BOOST_CHECK_CLOSE(childCounts.Get(2,0,RIGHT_CHILD), 7.5, 0.1);

    BOOST_CHECK_CLOSE(leftStats.Get(0,0,0), 2, 0.1);
    BOOST_CHECK_CLOSE(rightStats.Get(0,0,0), 1, 0.1);

    BOOST_CHECK_CLOSE(leftStats.Get(1,1,2), 1, 0.1);
    BOOST_CHECK_CLOSE(rightStats.Get(1,1,2), 0.5, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()