#include <boost/test/unit_test.hpp>

#include <vector>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "Constants.h"
#include "ForestPredictor.h"
#include "LinearMatrixFeature.h"
#include "ClassProbabilityCombiner.h"
#include "AllSamplesStep.h"


struct ForestPredictorFixture {
    ForestPredictorFixture()
    : xs_key("xs")
    , forest(2)
    , numberOfClasses(3)
    , indicesStep(xs_key)
    , feature(indicesStep.IndicesBufferId, xs_key)
    , combiner(numberOfClasses)
    , forestPredictor(NULL)
    {
        float xs_data[] = {4.0, 0.0};
        xs = MatrixBufferTemplate<float>(&xs_data[0], 1, 2);
        collection.AddBuffer(xs_key, xs);

        int depth_data[] = {0, 1, 1, 2, 2};
        depth = VectorBufferTemplate<int>(&depth_data[0], 5);

        float counts_data[] = {5, 5, 5, 5, 5};
        counts = VectorBufferTemplate<float>(&counts_data[0], 5);

        int path_1_data[] = {1, 2,
                         3, 4,
                        -1, -1,
                        -1, -1,
                        -1, -1};
        path_1 = MatrixBufferTemplate<int>(&path_1_data[0], 5, 2);
        int int_params_1_data[] = {MATRIX_FEATURES, 1, 0,
                        MATRIX_FEATURES, 1, 1,
                        MATRIX_FEATURES, 1, 0,
                        MATRIX_FEATURES, 1, 0,
                        MATRIX_FEATURES, 1, 0};
        int_params_1 = MatrixBufferTemplate<int>(&int_params_1_data[0], 5, 3);
        float float_params_1_data[] = {2.0, 0, 1.0,
                                        -5.0, 0, 1.0,
                                        0, 0, 0,
                                        0, 0, 0,
                                        0, 0, 0};
        float_params_1 = MatrixBufferTemplate<float>(&float_params_1_data[0], 5, 3);
        float estimator_params_1_data[] = {0, 0, 0,
                                            0, 0, 0,
                                            0.7,0.1,0.2,
                                            0.3,0.3,0.4,
                                            0.3,0.6,0.1 };
        estimator_params_1 = MatrixBufferTemplate<float>(&estimator_params_1_data[0], 5, 3);

        int path_2_data[] = {1, 2,
                             3, 4,
                            -1, -1,
                            -1, -1,
                            -1, -1};
        path_2 = MatrixBufferTemplate<int>(&path_2_data[0], 5, 2);
        int int_params_2_data[] = {MATRIX_FEATURES, 1, 0,
                                    MATRIX_FEATURES, 1, 0,
                                    MATRIX_FEATURES, 1, 0,
                                    MATRIX_FEATURES, 1, 0,
                                    MATRIX_FEATURES, 1, 0};
        int_params_2 = MatrixBufferTemplate<int>(&int_params_2_data[0], 5, 3);
        float float_params_2_data[] = {5.0, 0, 1.0,
                                        2.5, 0, 1.0,
                                        0, 0, 0,
                                        0, 0, 0,
                                        0, 0, 0};
        float_params_2 = MatrixBufferTemplate<float>(&float_params_2_data[0], 5, 3);
        float estimator_params_2_data[] = {0, 0, 0,
                                            0, 0, 0,
                                            0.8,0.1,0.1,
                                            0.2,0.2,0.6,
                                            0.2,0.7,0.1 };
        estimator_params_2 = MatrixBufferTemplate<float>(&estimator_params_2_data[0], 5, 3);

        forest.mTrees[0] = Tree(path_1, int_params_1, float_params_1, depth, counts, estimator_params_1);
        forest.mTrees[1] = Tree(path_2, int_params_2, float_params_2, depth, counts, estimator_params_2);

        forestPredictor = new TemplateForestPredictor< LinearMatrixFeature_t, ClassProbabilityCombiner<BufferTypes_t>, BufferTypes_t>(
                                forest, feature, combiner, &indicesStep);

    }

    ~ForestPredictorFixture()
    {
        delete forestPredictor;
    }

    VectorBufferTemplate<int> depth;
    VectorBufferTemplate<float> counts;

    MatrixBufferTemplate<int> path_1;
    MatrixBufferTemplate<int> int_params_1;
    MatrixBufferTemplate<float> float_params_1;
    MatrixBufferTemplate<float> estimator_params_1;

    MatrixBufferTemplate<int> path_2;
    MatrixBufferTemplate<int> int_params_2;
    MatrixBufferTemplate<float> float_params_2;
    MatrixBufferTemplate<float> estimator_params_2;

    Forest forest;

    BufferCollectionKey_t xs_key;
    MatrixBufferTemplate<float> xs;
    BufferCollection collection;

    typedef SinglePrecisionBufferTypes BufferTypes_t;
    typedef LinearMatrixFeature<BufferTypes_t, MatrixBufferTemplate<BufferTypes_t::SourceContinuous > > LinearMatrixFeature_t;

    int numberOfClasses;
    AllSamplesStep<BufferTypes_t, MatrixBufferTemplate<BufferTypes_t::SourceContinuous > > indicesStep;
    LinearMatrixFeature_t feature;
    ClassProbabilityCombiner<BufferTypes_t> combiner;
    TemplateForestPredictor< LinearMatrixFeature_t, ClassProbabilityCombiner<BufferTypes_t>, BufferTypes_t>* forestPredictor;
};

BOOST_FIXTURE_TEST_SUITE( ForestPredictorTests,  ForestPredictorFixture)

BOOST_AUTO_TEST_CASE(test_PredictLeafs)
{
    MatrixBufferTemplate<int> leafs;
    forestPredictor->PredictLeafs(collection, leafs);

    BOOST_CHECK_EQUAL(leafs.Get(0,0), 3);
    BOOST_CHECK_EQUAL(leafs.Get(0,1), 2);
}

BOOST_AUTO_TEST_CASE(test_PredictYs)
{
    MatrixBufferTemplate<float> ys;
    forestPredictor->PredictYs(collection, ys);

    BOOST_CHECK_CLOSE(ys.Get(0,0), 0.55, 0.1);
    BOOST_CHECK_CLOSE(ys.Get(0,1), 0.2, 0.1);
    BOOST_CHECK_CLOSE(ys.Get(0,2), 0.25, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()
