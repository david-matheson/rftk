#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferTypes.h"

#include "ClassProbabilityCombiner.h"

BOOST_AUTO_TEST_SUITE( ClassProbabilityCombinerTests )

BOOST_AUTO_TEST_CASE(test_Combine)
{
    float leaf_prob_data[] = {0.1, 0.9, 0,
                          0.3, 0.4, 0.3};
    MatrixBufferTemplate<float> leaf_prob(&leaf_prob_data[0], 2, 3);

    ClassProbabilityCombiner< SinglePrecisionBufferTypes > classProbabilityCombiner(leaf_prob.GetN());
    MatrixBufferTemplate<float> result(2,3);

    classProbabilityCombiner.Combine(0, 1.0, leaf_prob,  1.0);
    classProbabilityCombiner.WriteResult(0, result);
    float expected_result_1_data[] = {0.1, 0.9, 0,
                                    0.0, 0.0, 0.0};
    MatrixBufferTemplate<float> expected_1_result(&expected_result_1_data[0], 2, 3);
    BOOST_CHECK( result == expected_1_result );

    classProbabilityCombiner.Combine(1, 1.0, leaf_prob,  1.0);
    classProbabilityCombiner.WriteResult(1, result);
    float expected_result_2_data[] = {0.1, 0.9, 0,
                                    0.2, 0.65, 0.15};
    MatrixBufferTemplate<float> expected_2_result(&expected_result_2_data[0], 2, 3);
    BOOST_CHECK( result == expected_2_result );

    classProbabilityCombiner.Combine(1, 1.0, leaf_prob,  1.0);
    classProbabilityCombiner.WriteResult(0, result);
    float expected_result_3_data[] = {0.233333333, 0.56666666, 0.2,
                                    0.2, 0.65, 0.15};
    MatrixBufferTemplate<float> expected_3_result(&expected_result_3_data[0], 2, 3);
    BOOST_CHECK( result.AlmostEqual(expected_3_result) );

}

BOOST_AUTO_TEST_CASE(test_Reset)
{
    float leaf_prob_data[] = {0.1, 0.9, 0,
                          0.3, 0.4, 0.3};
    MatrixBufferTemplate<float> leaf_prob(&leaf_prob_data[0], 2, 3);

    ClassProbabilityCombiner< SinglePrecisionBufferTypes > classProbabilityCombiner(leaf_prob.GetN());
    MatrixBufferTemplate<float> result(2,3);

    classProbabilityCombiner.Combine(0, 1.0, leaf_prob,  1.0);
    classProbabilityCombiner.WriteResult(0, result);
    float expected_result_1_data[] = {0.1, 0.9, 0,
                                    0.0, 0.0, 0.0};
    MatrixBufferTemplate<float> expected_1_result(&expected_result_1_data[0], 2, 3);
    BOOST_CHECK( result == expected_1_result );

    classProbabilityCombiner.Reset();
    classProbabilityCombiner.WriteResult(0, result);
    float expected_result_2_data[] = {0.0, 0.0, 0,
                                    0.0, 0.0, 0.0};
    MatrixBufferTemplate<float> expected_2_result(&expected_result_2_data[0], 2, 3);
    BOOST_CHECK( result == expected_2_result );

    classProbabilityCombiner.Combine(0, 1.0, leaf_prob,  1.0);
    classProbabilityCombiner.WriteResult(0, result);
    float expected_result_3_data[] = {0.1, 0.9, 0,
                                    0.0, 0.0, 0.0};
    MatrixBufferTemplate<float> expected_3_result(&expected_result_3_data[0], 2, 3);
    BOOST_CHECK( result == expected_3_result );
}

BOOST_AUTO_TEST_SUITE_END()