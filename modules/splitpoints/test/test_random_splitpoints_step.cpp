#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "BufferTypes.h"
#include "UniqueBufferId.h"
#include "RandomSplitpointsStep.h"


BOOST_AUTO_TEST_SUITE(RandomSplitpointsStepTest)

BOOST_AUTO_TEST_CASE(test_ProcessStep_FEATURES_BY_DATAPOINTS)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId feature_values_key = "feature_values";
    float feature_values_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                   1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 0, 0, 0, 0 };
    MatrixBufferTemplate<float> feature_values(&feature_values_data[0], 3, 10);
    bc.AddBuffer(feature_values_key, feature_values);

    const int maxNumberOfSplitpoints = 4;
    RandomSplitpointsStep< BufferTypes<float, int, int, float, int, float, float, int, float> > randomSplitpointsStep(feature_values_key, maxNumberOfSplitpoints, FEATURES_BY_DATAPOINTS);
    boost::mt19937 gen;
    randomSplitpointsStep.ProcessStep(stack, bc, gen);

    MatrixBufferTemplate<float>& splitPoints = 
            bc.GetBuffer< MatrixBufferTemplate<float> >(randomSplitpointsStep.SplitpointsBufferId);
    VectorBufferTemplate<int>& splitPointsCounts = 
            bc.GetBuffer< VectorBufferTemplate<int> >(randomSplitpointsStep.SplitpointsCountsBufferId);

    BOOST_CHECK_EQUAL(splitPointsCounts.Get(0), maxNumberOfSplitpoints);
    BOOST_CHECK_EQUAL(splitPointsCounts.Get(1), 2);
    BOOST_CHECK_EQUAL(splitPointsCounts.Get(2), maxNumberOfSplitpoints);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_DATAPOINTS_BY_FEATURES)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId feature_values_key = "feature_values";
    float feature_values_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                   1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                   1.001, 1.002, 1.003, 1.004, 1.005, 1.006, 0, 0, 0, 0 };
    MatrixBufferTemplate<float> feature_values(&feature_values_data[0], 3, 10);
    bc.AddBuffer(feature_values_key, feature_values.Transpose());

    const int maxNumberOfSplitpoints = 4;
    RandomSplitpointsStep< BufferTypes<float, int, int, float, int, float, float, int, float> > randomSplitpointsStep(feature_values_key, maxNumberOfSplitpoints, DATAPOINTS_BY_FEATURES);
    boost::mt19937 gen;
    randomSplitpointsStep.ProcessStep(stack, bc, gen);

    MatrixBufferTemplate<float>& splitPoints = 
            bc.GetBuffer< MatrixBufferTemplate<float> >(randomSplitpointsStep.SplitpointsBufferId);
    VectorBufferTemplate<int>& splitPointsCounts = 
            bc.GetBuffer< VectorBufferTemplate<int> >(randomSplitpointsStep.SplitpointsCountsBufferId);

    BOOST_CHECK_EQUAL(splitPointsCounts.Get(0), maxNumberOfSplitpoints);
    BOOST_CHECK_EQUAL(splitPointsCounts.Get(1), 2);
    BOOST_CHECK_EQUAL(splitPointsCounts.Get(2), maxNumberOfSplitpoints);
}

BOOST_AUTO_TEST_CASE(test_ProcessStep_SEQUENTUAL)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId feature_values_key = "feature_values";
    float feature_values1_data[] = {1.0, 2.0, 3.0, 4.0, 5.0,
                                   1.0, 1.0, 1.0, 1.0, 1.0, 
                                   1.001, 1.002, 1.003, 1.004, 1.005 };
    MatrixBufferTemplate<float> feature_values1(&feature_values1_data[0], 3, 5);
    bc.AddBuffer(feature_values_key, feature_values1);

    boost::mt19937 gen;
    const int maxNumberOfSplitpoints = 5;
    RandomSplitpointsStep< BufferTypes<float, int, int, float, int, float, float, int, float> > randomSplitpointsStep(feature_values_key, maxNumberOfSplitpoints, FEATURES_BY_DATAPOINTS);

    randomSplitpointsStep.ProcessStep(stack, bc, gen);
    MatrixBufferTemplate<float> splitPoints1 = 
            bc.GetBuffer< MatrixBufferTemplate<float> >(randomSplitpointsStep.SplitpointsBufferId);
    VectorBufferTemplate<int> splitPointsCounts1 = 
            bc.GetBuffer< VectorBufferTemplate<int> >(randomSplitpointsStep.SplitpointsCountsBufferId);

    BOOST_CHECK_EQUAL(splitPointsCounts1.Get(0), maxNumberOfSplitpoints);
    BOOST_CHECK_EQUAL(splitPointsCounts1.Get(1), 1);
    BOOST_CHECK_EQUAL(splitPointsCounts1.Get(2), maxNumberOfSplitpoints);

    float feature_values2_data[] = {6.0, 7.0, 8.0, 9.0, 10.0,
                                   2.0, 2.0, 2.0, 2.0, 2.0,
                                   1.006, 0, 0, 0, 0 };
    MatrixBufferTemplate<float> feature_values2(&feature_values2_data[0], 3, 5);
    bc.AddBuffer(feature_values_key, feature_values2);
    randomSplitpointsStep.ProcessStep(stack, bc, gen);
    MatrixBufferTemplate<float> splitPoints2 = 
            bc.GetBuffer< MatrixBufferTemplate<float> >(randomSplitpointsStep.SplitpointsBufferId);
    VectorBufferTemplate<int> splitPointsCounts2 = 
            bc.GetBuffer< VectorBufferTemplate<int> >(randomSplitpointsStep.SplitpointsCountsBufferId);

    BOOST_CHECK_EQUAL(splitPointsCounts2.Get(0), maxNumberOfSplitpoints);
    BOOST_CHECK_EQUAL(splitPointsCounts2.Get(1), 2);
    BOOST_CHECK_EQUAL(splitPointsCounts2.Get(2), maxNumberOfSplitpoints);
    BOOST_CHECK(splitPoints1.SliceRow(0) == splitPoints2.SliceRow(0));
    BOOST_CHECK(splitPoints1.SliceRow(2) == splitPoints2.SliceRow(2));
}

BOOST_AUTO_TEST_SUITE_END()