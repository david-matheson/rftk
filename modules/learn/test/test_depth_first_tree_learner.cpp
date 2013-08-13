#include <boost/test/unit_test.hpp>

#include "CreateDepthFirstLearner.h"


BOOST_FIXTURE_TEST_SUITE( DepthFirstTreeLearnerTests,  DepthFirstTreeLearnerFixture )

BOOST_AUTO_TEST_CASE(test_Learn)
{
    const int numberOfClasses = 4;
    const double minNodeSize = 1.0;
    FeatureValueOrdering featureOrdering = FEATURES_BY_DATAPOINTS;

    DepthFirstTreeLearner<CdflBufferTypes_t> depthFirstTreeLearner = CreateDepthFirstLearner(xs_key, classes_key, numberOfClasses, featureOrdering, minNodeSize);
    Tree tree(1, 3, 3, numberOfClasses );
    depthFirstTreeLearner.Learn(stack, tree, 0);

    int expected_path_data[] = { 1,2,
                        -1,-1,
                        3,4,
                        5,6,
                        -1,-1,
                        -1,-1,
                        -1,-1,
                        -1,-1};
    MatrixBufferTemplate<int> expected_path(&expected_path_data[0], 8, 2);
    BOOST_CHECK( tree.mPath == expected_path );

    int expected_int_params_data[] = { MATRIX_FEATURES, 1, 0,
                                      0,0,0,
                                      MATRIX_FEATURES, 1, 1,
                                      MATRIX_FEATURES, 1, 1,
                                      0,0,0,
                                      0,0,0,
                                      0,0,0,
                                      0,0,0};
    MatrixBufferTemplate<int> expected_int_params(&expected_int_params_data[0], 8, 3);
    BOOST_CHECK( tree.mIntFeatureParams == expected_int_params );

    float expected_float_params_data[] = { 2, 0, 1,
                                          0,0,0,
                                          -1.5, 0, 1,
                                          1, 0, 1,
                                          0,0,0,
                                          0,0,0,
                                          0,0,0,
                                          0,0,0};
    MatrixBufferTemplate<float> expected_float_params(&expected_float_params_data[0], 8, 3);
    BOOST_CHECK( tree.mFloatFeatureParams == expected_float_params );

    float expected_counts_data[] = { 0, 5, 5, 3, 2, 2, 1, 0 };
    VectorBufferTemplate<float> expected_counts(&expected_counts_data[0], 8);
    BOOST_CHECK( tree.mCounts == expected_counts );

    int expected_depths_data[] = { 0, 1, 1, 2, 2, 3, 3, 0 };
    VectorBufferTemplate<int> expected_depths(&expected_depths_data[0], 8);
    BOOST_CHECK( tree.mDepths == expected_depths );

    BOOST_CHECK_CLOSE( tree.mYs.Get(1,0), 1, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,1), 0.4, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,2), 0.4, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,3), 0.2, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(3,2), 0.6666, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(3,3), 0.3333, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(4,1), 1.0, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(5,2), 1.0, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(6,3), 1.0, 0.1 );
}

BOOST_AUTO_TEST_CASE(test_Learn_DATAPOINTS_BY_FEATURES)
{
    const int numberOfClasses = 4;
    const double minNodeSize = 1.0;
    FeatureValueOrdering featureOrdering = DATAPOINTS_BY_FEATURES;

    DepthFirstTreeLearner<CdflBufferTypes_t> depthFirstTreeLearner = CreateDepthFirstLearner(xs_key, classes_key, numberOfClasses, featureOrdering, minNodeSize);
    Tree tree(1, 4, 3, numberOfClasses );
    depthFirstTreeLearner.Learn(stack, tree, 0);

    int expected_path_data[] = { 1,2,
                        -1,-1,
                        3,4,
                        5,6,
                        -1,-1,
                        -1,-1,
                        -1,-1,
                        -1,-1};
    MatrixBufferTemplate<int> expected_path(&expected_path_data[0], 8, 2);
    BOOST_CHECK( tree.mPath == expected_path );

    int expected_int_params_data[] = { MATRIX_FEATURES, 1, 0, 0,
                                      0,0,0,0,
                                      MATRIX_FEATURES, 1, 1, 0,
                                      MATRIX_FEATURES, 1, 1, 0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      0,0,0,0};
    MatrixBufferTemplate<int> expected_int_params(&expected_int_params_data[0], 8, 4);
    BOOST_CHECK( tree.mIntFeatureParams == expected_int_params );

    float expected_float_params_data[] = { 2, 0, 1,
                                          0,0,0,
                                          -1.5, 0, 1,
                                          1, 0, 1,
                                          0,0,0,
                                          0,0,0,
                                          0,0,0,
                                          0,0,0};
    MatrixBufferTemplate<float> expected_float_params(&expected_float_params_data[0], 8, 3);
    BOOST_CHECK( tree.mFloatFeatureParams == expected_float_params );

    float expected_counts_data[] = { 0, 5, 5, 3, 2, 2, 1, 0 };
    VectorBufferTemplate<float> expected_counts(&expected_counts_data[0], 8);
    BOOST_CHECK( tree.mCounts == expected_counts );

    int expected_depths_data[] = { 0, 1, 1, 2, 2, 3, 3, 0 };
    VectorBufferTemplate<int> expected_depths(&expected_depths_data[0], 8);
    BOOST_CHECK( tree.mDepths == expected_depths );

    BOOST_CHECK_CLOSE( tree.mYs.Get(1,0), 1, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,1), 0.4, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,2), 0.4, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,3), 0.2, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(3,2), 0.6666, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(3,3), 0.3333, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(4,1), 1.0, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(5,2), 1.0, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(6,3), 1.0, 0.1 );
}

BOOST_AUTO_TEST_CASE(test_Learn_minsize)
{
    // Constants
    const int numberOfClasses = 4;
    FeatureValueOrdering featureOrdering = FEATURES_BY_DATAPOINTS;
    const double minNodeSize = 6.0;

    DepthFirstTreeLearner<CdflBufferTypes_t> depthFirstTreeLearner = CreateDepthFirstLearner(xs_key, classes_key, numberOfClasses, featureOrdering, minNodeSize);
    Tree tree(1, 3, 3, numberOfClasses );
    depthFirstTreeLearner.Learn(stack, tree, 0);

    int expected_path_data[] = { 1,2,
                        -1,-1,
                        -1,-1,
                        -1,-1,
                        -1,-1 };
    MatrixBufferTemplate<int> expected_path(&expected_path_data[0], 5, 2);
    BOOST_CHECK( tree.mPath == expected_path );

    int expected_int_params_data[] = { MATRIX_FEATURES, 1, 0,
                                      0,0,0,
                                      0,0,0,
                                      0,0,0,
                                      0,0,0
                                      };
    MatrixBufferTemplate<int> expected_int_params(&expected_int_params_data[0], 5, 3);
    BOOST_CHECK( tree.mIntFeatureParams == expected_int_params );

    float expected_float_params_data[] = { 2, 0, 1,
                                          0,0,0,
                                          0,0,0,
                                          0,0,0,
                                          0,0,0};
    MatrixBufferTemplate<float> expected_float_params(&expected_float_params_data[0], 5, 3);
    BOOST_CHECK( tree.mFloatFeatureParams == expected_float_params );

    float expected_counts_data[] = { 0, 5, 5, 0, 0};
    VectorBufferTemplate<float> expected_counts(&expected_counts_data[0], 5);
    BOOST_CHECK( tree.mCounts == expected_counts );

    int expected_depths_data[] = { 0, 1, 1, 0, 0 };
    VectorBufferTemplate<int> expected_depths(&expected_depths_data[0], 5);
    BOOST_CHECK( tree.mDepths == expected_depths );

    BOOST_CHECK_CLOSE( tree.mYs.Get(1,0), 1, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,1), 0.4, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,2), 0.4, 0.1 );
    BOOST_CHECK_CLOSE( tree.mYs.Get(2,3), 0.2, 0.1 );
}

BOOST_AUTO_TEST_SUITE_END()