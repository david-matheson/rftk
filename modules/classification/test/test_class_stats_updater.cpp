#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "ClassStatsUpdater.h"

BOOST_AUTO_TEST_SUITE( ClassStatsUpdaterTests )

BOOST_AUTO_TEST_CASE(test_UpdateStats)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId weights_key = "weights";
    float weight_data[] = {1.0, 0.5, 2.0, 0.0, 3.0};
    VectorBufferTemplate<float> weights(&weight_data[0], 5);
    bc.AddBuffer(weights_key, weights);

    BufferId classes_key = "classes";
    int classes_data[] = {0, 2, 1, 2, 1};
    VectorBufferTemplate<int> classes(&classes_data[0], 5);
    bc.AddBuffer(classes_key, classes);

    const int numberOfClasses = 3;
    ClassStatsUpdater<float, int> classStatsUpdater(weights_key, classes_key, numberOfClasses);
    BindedClassStatsUpdater<float, int> bindedClassStatsUpdater = classStatsUpdater.Bind(stack);

    float counts;
    Tensor3BufferTemplate<float> stats(4,5,numberOfClasses);

    bindedClassStatsUpdater.UpdateStats(counts, stats, 0,0,0);
    BOOST_CHECK_CLOSE(counts, 1.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,0), 1.0, 0.1);

    bindedClassStatsUpdater.UpdateStats(counts, stats, 2,3,1);
    BOOST_CHECK_CLOSE(counts, 1.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(2,3,2), 0.5, 0.1);

    bindedClassStatsUpdater.UpdateStats(counts, stats, 0,0,2);
    BOOST_CHECK_CLOSE(counts, 3.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,1), 2.0, 0.1);

    bindedClassStatsUpdater.UpdateStats(counts, stats, 0,0,3);
    BOOST_CHECK_CLOSE(counts, 3.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,2), 0.0, 0.1);

    bindedClassStatsUpdater.UpdateStats(counts, stats, 0,0,4);
    BOOST_CHECK_CLOSE(counts, 6.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,1), 5.0, 0.1);
}


BOOST_AUTO_TEST_SUITE_END()