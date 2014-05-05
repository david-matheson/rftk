#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "BestSplitpointsWalkingSortedStep.h"
#include "TestBufferWalker.h"

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
    T data[] = {0.06, 0.03, 0.07, 0,    0.02,
                6.1,  3,    5.5,  8.9,  2, 
                9,    3,    6.25, 8.75, 8.8};

    return MatrixBufferTemplate<T>(&data[0], 3, 5);
}

template<typename T>
MatrixBufferTemplate<T> CreateImpurityMatrix()
{
    T data[] = {0.01, 0.02, 0.05, 0.2, 0.04,
                50, 8, 6, 4, 2, 
                20, 10, 30, 20, 40,
                7, 4, 66, 33, 2,
                55, 6, -8, 2, 3};

    return MatrixBufferTemplate<T>(&data[0], 5, 5);
}

template<typename T>
MatrixBufferTemplate<T> CreateLeftYsMatrix()
{
    T data[] = {1, 2, 3, 4, 5,
                10, 8, 6, 4, 2, 
                20, 10, 30, 20, 40,
                7, 4, 66, 33, 2,
                55, 6, -8, 2, 3};

    return MatrixBufferTemplate<T>(&data[0], 5, 5);
}

template<typename T>
MatrixBufferTemplate<T> CreateRightYsMatrix()
{
    T data[] = {10, 8, 6, 4, 2, 
                20, 10, 30, 20, 40,
                7, 4, 66, 33, 2,
                55, 6, -8, 2, 3,
                1, 2, 3, 4, 5,};

    return MatrixBufferTemplate<T>(&data[0], 5, 5);
}

struct BestSplitpointsWalkingSortedStepFixture {
    BestSplitpointsWalkingSortedStepFixture()
    : fm_key("feature")
    , fm( CreateFeatureMatrix<double>() )
    , im( CreateImpurityMatrix<double>() )
    , left( CreateLeftYsMatrix<double>() )
    , right( CreateRightYsMatrix<double>() )
    , collection()
    , stack()
    {
        collection.AddBuffer(fm_key, fm);
        stack.Push(&collection);
    }

    ~BestSplitpointsWalkingSortedStepFixture()
    {
    }

    const BufferCollectionKey_t fm_key;
    const MatrixBufferTemplate<double> fm;
    const MatrixBufferTemplate<double> im;    
    const MatrixBufferTemplate<double> left;
    const MatrixBufferTemplate<double> right;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( BestSplitpointsWalkingSortedStepTests,  BestSplitpointsWalkingSortedStepFixture )


BOOST_AUTO_TEST_CASE(test_BestSplitpointsWalkingSortedStep_ProcessStep_FEATURES_BY_DATAPOINTS)
{
    BOOST_CHECK( fm == stack.GetBuffer< MatrixBufferTemplate<double> >(fm_key) );
    TestBufferWalker<double, int> walker(im, left, right);
    BestSplitpointsWalkingSortedStep< TestBufferWalker<double, int> > bestsplits(walker, fm_key, FEATURES_BY_DATAPOINTS, AT_MIDPOINT);
    boost::mt19937 gen;
    bestsplits.ProcessStep(stack, collection, gen, collection, 0);

    BOOST_CHECK( collection.HasBuffer< MatrixBufferTemplate<double> >( bestsplits.ImpurityBufferId ) );
    MatrixBufferTemplate<double>& best_impurity = collection.GetBuffer< MatrixBufferTemplate<double> >( bestsplits.ImpurityBufferId );
    double expected_best_impurity_data[] = {0.2, 50, 40};
    MatrixBufferTemplate<double> expected_best_impurity = CreateMatrix<double>(expected_best_impurity_data, 3, 1);
    BOOST_CHECK(best_impurity == expected_best_impurity);

    BOOST_CHECK( collection.HasBuffer< MatrixBufferTemplate<double> >( bestsplits.SplitpointBufferId ) );
    MatrixBufferTemplate<double>& best_thresholds = collection.GetBuffer< MatrixBufferTemplate<double> >( bestsplits.SplitpointBufferId );
    double expected_best_thresholds_data[] = {0.01, 7.5, 8.9};
    MatrixBufferTemplate<double> expected_best_thresholds = CreateMatrix<double>(expected_best_thresholds_data, 3, 1);
    BOOST_CHECK(best_thresholds == expected_best_thresholds);

    BOOST_CHECK( collection.HasBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId ) );
    VectorBufferTemplate<int>& threshold_counts = collection.GetBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId );
    int expected_threshold_counts_data[] = {1, 1, 1};
    VectorBufferTemplate<int> expected_threshold_counts = CreateVector<int>(expected_threshold_counts_data, 3);
    BOOST_CHECK(threshold_counts == expected_threshold_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<double> >( bestsplits.ChildCountsBufferId ) );
    Tensor3BufferTemplate<double>& child_counts = collection.GetBuffer< Tensor3BufferTemplate<double> >( bestsplits.ChildCountsBufferId );
    double expected_child_counts_data[] = {15, 30,
                                          30, 120,
                                          120, 112};
    Tensor3BufferTemplate<double> expected_child_counts = CreateTensor3<double>(expected_child_counts_data, 3, 1, 2);
    BOOST_CHECK(child_counts == expected_child_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<double> >( bestsplits.LeftYsBufferId ) );
    Tensor3BufferTemplate<double>& left_ys = collection.GetBuffer< Tensor3BufferTemplate<double> >( bestsplits.LeftYsBufferId );
    double expected_left_ys_data[] = {1, 2, 3, 4, 5,
                                      10, 8, 6, 4, 2, 
                                      20, 10, 30, 20, 40};
    Tensor3BufferTemplate<double> expected_left_ys = CreateTensor3<double>(expected_left_ys_data, 3, 1, 5);
    BOOST_CHECK(left_ys == expected_left_ys);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<double> >( bestsplits.RightYsBufferId ) );
    Tensor3BufferTemplate<double>& right_ys = collection.GetBuffer< Tensor3BufferTemplate<double> >( bestsplits.RightYsBufferId );
    double expected_right_ys_data[] = {10, 8, 6, 4, 2, 
                                      20, 10, 30, 20, 40,
                                      7, 4, 66, 33, 2};
    Tensor3BufferTemplate<double> expected_right_ys = CreateTensor3<double>(expected_right_ys_data, 3, 1, 5);
    BOOST_CHECK(right_ys == expected_right_ys);
}

BOOST_AUTO_TEST_CASE(test_BestSplitpointsWalkingSortedStep_ProcessStep_DATAPOINTS_BY_FEATURES)
{
    BOOST_CHECK( fm == stack.GetBuffer< MatrixBufferTemplate<double> >(fm_key) );
    TestBufferWalker<double, int> walker(im, left, right);
    BestSplitpointsWalkingSortedStep< TestBufferWalker<double, int> > bestsplits(walker, fm_key, DATAPOINTS_BY_FEATURES, AT_MIDPOINT);
    boost::mt19937 gen;
    bestsplits.ProcessStep(stack, collection, gen, collection, 0);

    BOOST_CHECK( collection.HasBuffer< MatrixBufferTemplate<double> >( bestsplits.ImpurityBufferId ) );
    MatrixBufferTemplate<double>& best_impurity = collection.GetBuffer< MatrixBufferTemplate<double> >( bestsplits.ImpurityBufferId );
    double expected_best_impurity_data[] = {0.02, 50, 20, 66, 55};
    MatrixBufferTemplate<double> expected_best_impurity = CreateMatrix<double>(expected_best_impurity_data, 5, 1);
    BOOST_CHECK(best_impurity == expected_best_impurity);

    BOOST_CHECK( collection.HasBuffer< MatrixBufferTemplate<double> >( bestsplits.SplitpointBufferId ) );
    MatrixBufferTemplate<double>& best_thresholds = collection.GetBuffer< MatrixBufferTemplate<double> >( bestsplits.SplitpointBufferId );
    double expected_best_thresholds_data[] = {7.55, 1.515, 2.785, 8.825, 1.01};
    MatrixBufferTemplate<double> expected_best_thresholds = CreateMatrix<double>(expected_best_thresholds_data, 5, 1);
    BOOST_CHECK_CLOSE(best_thresholds.Get(0,0), expected_best_thresholds.Get(0,0), 0.001);
    BOOST_CHECK_CLOSE(best_thresholds.Get(1,0), expected_best_thresholds.Get(1,0), 0.001);
    BOOST_CHECK_CLOSE(best_thresholds.Get(2,0), expected_best_thresholds.Get(2,0), 0.001);
    BOOST_CHECK_CLOSE(best_thresholds.Get(3,0), expected_best_thresholds.Get(3,0), 0.001);
    BOOST_CHECK_CLOSE(best_thresholds.Get(4,0), expected_best_thresholds.Get(4,0), 0.001);
    // BOOST_CHECK(best_thresholds == expected_best_thresholds);

    BOOST_CHECK( collection.HasBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId ) );
    VectorBufferTemplate<int>& threshold_counts = collection.GetBuffer< VectorBufferTemplate<int> >( bestsplits.SplitpointCountsBufferId );
    int expected_threshold_counts_data[] = {1, 1, 1, 1, 1};
    VectorBufferTemplate<int> expected_threshold_counts = CreateVector<int>(expected_threshold_counts_data, 5);
    BOOST_CHECK(threshold_counts == expected_threshold_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<double> >( bestsplits.ChildCountsBufferId ) );
    Tensor3BufferTemplate<double>& child_counts = collection.GetBuffer< Tensor3BufferTemplate<double> >( bestsplits.ChildCountsBufferId );
    double expected_child_counts_data[] = {15, 30,
                                          30, 120,
                                          120, 112,
                                          112, 58,
                                          58, 15};
    Tensor3BufferTemplate<double> expected_child_counts = CreateTensor3<double>(expected_child_counts_data, 5, 1, 2);
    BOOST_CHECK(child_counts == expected_child_counts);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<double> >( bestsplits.LeftYsBufferId ) );
    Tensor3BufferTemplate<double>& left_ys = collection.GetBuffer< Tensor3BufferTemplate<double> >( bestsplits.LeftYsBufferId );
    double expected_left_ys_data[] = {1, 2, 3, 4, 5,
                                      10, 8, 6, 4, 2, 
                                      20, 10, 30, 20, 40,
                                      7, 4, 66, 33, 2,
                                      55, 6, -8, 2, 3};
    Tensor3BufferTemplate<double> expected_left_ys = CreateTensor3<double>(expected_left_ys_data, 5, 1, 5); 
    BOOST_CHECK(left_ys == expected_left_ys);

    BOOST_CHECK( collection.HasBuffer< Tensor3BufferTemplate<double> >( bestsplits.RightYsBufferId ) );
    Tensor3BufferTemplate<double>& right_ys = collection.GetBuffer< Tensor3BufferTemplate<double> >( bestsplits.RightYsBufferId );
    double expected_right_ys_data[] = {10, 8, 6, 4, 2, 
                                      20, 10, 30, 20, 40,
                                      7, 4, 66, 33, 2,
                                      55, 6, -8, 2, 3,
                                      1, 2, 3, 4, 5};
    Tensor3BufferTemplate<double> expected_right_ys = CreateTensor3<double>(expected_right_ys_data, 5, 1, 5);
    BOOST_CHECK(right_ys == expected_right_ys);
}

BOOST_AUTO_TEST_SUITE_END()