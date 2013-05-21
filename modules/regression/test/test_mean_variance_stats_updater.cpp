#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "MeanVarianceStatsUpdater.h"

BOOST_AUTO_TEST_SUITE( MeanVarianceStatsUpdaterTests )

BOOST_AUTO_TEST_CASE(test_UpdateStats)
{
    BufferCollection bc;
    BufferCollectionStack stack;
    stack.Push(&bc);

    BufferId weights_key = "weights";
    float weight_data[] = {1.0, 0.5, 2.0, 0.0, 3.0};
    VectorBufferTemplate<float> weights(&weight_data[0], 5);
    bc.AddBuffer(weights_key, weights);

    BufferId ys_key = "ys";
    int ys_data[] = {0, 2.0, 
                    2, 1,
                    4, 5,
                    22, 24,
                    1, 2};
    MatrixBufferTemplate<float> ys(&ys_data[0], 5,2);
    bc.AddBuffer(ys_key, ys);

    const int dimensionOfY = ys.GetN();
    MeanVarianceStatsUpdater<float, int> meanVarianceStatsUpdater(weights_key, ys_key, dimensionOfY);
    BindedMeanVarianceStatsUpdater<float, int> bindedMeanVarianceStatsUpdater = meanVarianceStatsUpdater.Bind(stack);

    float counts;
    Tensor3BufferTemplate<float> stats(4,5,dimensionOfY*2);

    bindedMeanVarianceStatsUpdater.UpdateStats(counts, stats, 0,0,0);
    BOOST_CHECK_CLOSE(counts, 1.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,0), 0.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,1), 2.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,2), 0.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,3), 4.0, 0.1);

    bindedMeanVarianceStatsUpdater.UpdateStats(counts, stats, 2,3,1);
    BOOST_CHECK_CLOSE(counts, 1.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(2,3,0), 1, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(2,3,1), 0.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(2,3,2), 2, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(2,3,3), 0.5, 0.1);

    bindedMeanVarianceStatsUpdater.UpdateStats(counts, stats, 0,0,2);
    BOOST_CHECK_CLOSE(counts, 3.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,0), 8.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,1), 12.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,2), 32.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,3), 54.0, 0.1);

    bindedMeanVarianceStatsUpdater.UpdateStats(counts, stats, 0,0,3);
    BOOST_CHECK_CLOSE(counts, 3.5, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,0), 8.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,1), 12.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,2), 32.0, 0.1);
    BOOST_CHECK_CLOSE(stats.Get(0,0,3), 54.0, 0.1);
}


BOOST_AUTO_TEST_SUITE_END()