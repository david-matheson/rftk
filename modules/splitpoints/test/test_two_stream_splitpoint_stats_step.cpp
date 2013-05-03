#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"
#include "TwoStreamSplitpointStatsStep.h"
#include "ClassStatsUpdater.h"

BOOST_AUTO_TEST_SUITE(TwoStreamSplitpointStatsStepTest)

BOOST_AUTO_TEST_CASE(test_ProcessStep_FEATURES_BY_DATAPOINTS)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId weights_key = "weights";
    //int steam_type_dat = {1,  1,    0,   1,   0,   0,   0,  1,   0,   1};
    //int classes_data[] = {0,   2,   1,   2,   1,   0,   0,  1,   1,    2};
    float weight_data[] = {1.0, 0.5, 2.0, 0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    VectorBufferTemplate<float> weights(&weight_data[0], 10);
    bc.AddBuffer(weights_key, weights);

    BufferId classes_key = "classes";
    int classes_data[] = {0, 2, 1, 2, 1, 0, 0, 1, 1, 2};
    VectorBufferTemplate<int> classes(&classes_data[0], 10);
    bc.AddBuffer(classes_key, classes);

    const int numberOfClasses = 3;
    ClassStatsUpdater<float, int> classStatsUpdater(weights_key, classes_key, numberOfClasses);

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

    BufferId stream_type_key = "stream_type";
    int steam_type_data[] = {STREAM_IMPURITY,STREAM_IMPURITY,STREAM_ESTIMATION,STREAM_IMPURITY,
                              STREAM_ESTIMATION,STREAM_ESTIMATION,STREAM_ESTIMATION,STREAM_IMPURITY,
                              STREAM_ESTIMATION,STREAM_IMPURITY};
    VectorBufferTemplate<int> steam_type(&steam_type_data[0], 10);
    bc.AddBuffer(stream_type_key, steam_type);

    BufferId feature_values_key = "feature_values";
    float feature_values_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                   1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 0, 0, 0, 0 };
    MatrixBufferTemplate<float> feature_values(&feature_values_data[0], 3, 10);
    bc.AddBuffer(feature_values_key, feature_values);

    TwoStreamSplitpointStatsStep< ClassStatsUpdater<float, int> > splitpointStatsStep(split_points_key, 
                                                                              split_points_counts_key,
                                                                              stream_type_key,
                                                                              feature_values_key, 
                                                                              FEATURES_BY_DATAPOINTS,
                                                                              classStatsUpdater);
    boost::mt19937 gen;
    splitpointStatsStep.ProcessStep(stack, bc, gen);

    Tensor3BufferTemplate<float>& childImpurityCounts = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.ChildCountsImpurityBufferId);

    Tensor3BufferTemplate<float>& leftImpurityStats = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.LeftImpurityStatsBufferId);

    Tensor3BufferTemplate<float>& rightImpurityStats = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.RightImpurityStatsBufferId);

    Tensor3BufferTemplate<float>& childEstimatorCounts = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.ChildCountsEstimatorBufferId);

    Tensor3BufferTemplate<float>& leftEstimatorStats = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.LeftEstimatorStatsBufferId);

    Tensor3BufferTemplate<float>& rightEstimatorStats = 
            bc.GetBuffer< Tensor3BufferTemplate<float> >(splitpointStatsStep.RightEstimatorStatsBufferId);


    BOOST_CHECK_CLOSE(childImpurityCounts.Get(0,1,LEFT_CHILD), 2, 0.1);
    BOOST_CHECK_CLOSE(childImpurityCounts.Get(0,1,RIGHT_CHILD), 1.5, 0.1);

    BOOST_CHECK_CLOSE(childEstimatorCounts.Get(0,1,LEFT_CHILD), 3, 0.1);
    BOOST_CHECK_CLOSE(childEstimatorCounts.Get(0,1,RIGHT_CHILD), 5, 0.1);

    BOOST_CHECK_CLOSE(leftImpurityStats.Get(0,0,0), 0, 0.1);
    BOOST_CHECK_CLOSE(rightImpurityStats.Get(0,0,0), 1, 0.1);

    BOOST_CHECK_CLOSE(leftEstimatorStats.Get(0,0,0), 2, 0.1);
    BOOST_CHECK_CLOSE(rightEstimatorStats.Get(0,0,0), 0, 0.1);

    BOOST_CHECK_CLOSE(leftImpurityStats.Get(0,0,2), 1, 0.1);
    BOOST_CHECK_CLOSE(rightImpurityStats.Get(0,0,2), 0.5, 0.1);

    BOOST_CHECK_CLOSE(leftEstimatorStats.Get(0,1,1), 1, 0.1);
    BOOST_CHECK_CLOSE(rightEstimatorStats.Get(0,1,1), 5, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()