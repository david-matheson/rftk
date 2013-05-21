#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "SumOfVarianceWalker.h"
#include "BestSplitpointsWalkingSortedStep.h"


struct SumOfVarianceWalkerFixture {
    SumOfVarianceWalkerFixture()
    : fm_key("feature")
    , ys_key("ys")
    , weights_key("weights")
    , ydim(2)
    , collection()
    , stack()
    {
        float fm_data[] = {0.06, 0.03, 0.07, 0.01, 0.02, -0.15, -0.25, 0,
                            8.9,  3000,    2, 6.1,    2,    -9,     2,  2};
        const MatrixBufferTemplate<float> fm = MatrixBufferTemplate<float>(&fm_data[0], 2, 8);
        collection.AddBuffer(fm_key, fm);

        float ys_data[] = {0, 1,
                            0, 2,
                            0, 3,
                            0, 2,
                            5, 8,
                            6, 7,
                            4, 3,
                            4, 2};
        const MatrixBufferTemplate<float> ys = MatrixBufferTemplate<float>(&ys_data[0], 8, 2);
        collection.AddBuffer(ys_key, ys);

        float weights_data[] = {2,1,1,1,2,1,1,3};
        const VectorBufferTemplate<float> weights = VectorBufferTemplate<float>(&weights_data[0], 8);
        collection.AddBuffer(weights_key, weights);

        stack.Push(&collection);
    }

    ~SumOfVarianceWalkerFixture()
    {
    }

    const BufferCollectionKey_t fm_key;
    const BufferCollectionKey_t ys_key;
    const BufferCollectionKey_t weights_key;
    const int ydim;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( SumOfVarianceWalkerTests,  SumOfVarianceWalkerFixture )

BOOST_AUTO_TEST_CASE(test_SumOfVarianceWalker_Impurity)
{
    SumOfVarianceWalker<float, int> sumOfVarianceWalker(weights_key, ys_key, ydim);
    sumOfVarianceWalker.Bind(stack);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.0, 1);

    sumOfVarianceWalker.MoveLeftToRight(2);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.6622475, 1);

    sumOfVarianceWalker.MoveLeftToRight(5);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.5236111, 1);

    sumOfVarianceWalker.MoveLeftToRight(7);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.6557540, 1);

    sumOfVarianceWalker.MoveLeftToRight(3);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.2847214, 1);

    sumOfVarianceWalker.MoveLeftToRight(6);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.5605164, 1);

    sumOfVarianceWalker.MoveLeftToRight(0);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 2.3726845, 1);

    sumOfVarianceWalker.MoveLeftToRight(1);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 5.2902780, 1);

    sumOfVarianceWalker.MoveLeftToRight(4);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.0, 1);
}

BOOST_AUTO_TEST_CASE(test_SumOfVarianceWalker_Reset)
{
    SumOfVarianceWalker<float, int> sumOfVarianceWalker(weights_key, ys_key, ydim);
    sumOfVarianceWalker.Bind(stack);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.0, 1);

    sumOfVarianceWalker.MoveLeftToRight(2);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.6622475, 1);

    sumOfVarianceWalker.Reset();
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.0, 1);

    sumOfVarianceWalker.MoveLeftToRight(2);
    BOOST_CHECK_CLOSE(sumOfVarianceWalker.Impurity(), 0.6622475, 1);
}


BOOST_AUTO_TEST_SUITE_END()