#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "ClassEstimatorFinalizer.h"

BOOST_AUTO_TEST_SUITE( ClassEstimatorFinalizerTests )

typedef BufferTypes<float, int, int, float, int, float, double, int, float> ClassEstimatorFinalizerTestsBuffers_t;
typedef BufferTypes<double, int, int, float, int, float, double, int, float> ClassEstimatorFinalizerTestsBuffers2_t;

BOOST_AUTO_TEST_CASE(test_Finalize)
{
    double estimator_data[] = {0, 1.0, 2.0, 3.0, 4.0};
    VectorBufferTemplate<double> estimator(&estimator_data[0], 5);

    ClassEstimatorFinalizer<ClassEstimatorFinalizerTestsBuffers_t> classEstimatorFinalizer;
    VectorBufferTemplate<float> final_estimator = classEstimatorFinalizer.Finalize(estimator.Sum(), estimator);

    float expected_result_data[] = {0, 0.1, 0.2, 0.3, 0.4};
    VectorBufferTemplate<float> expected_result(&expected_result_data[0], 5);

    BOOST_CHECK( final_estimator == expected_result );
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    double estimator_data[] = {0, 1.0, 2.0, 3.0, 4.0};
    VectorBufferTemplate<double> estimator(&estimator_data[0], 5);

    FinalizerI<ClassEstimatorFinalizerTestsBuffers2_t>* classEstimatorFinalizer = new ClassEstimatorFinalizer<ClassEstimatorFinalizerTestsBuffers2_t>();
    FinalizerI<ClassEstimatorFinalizerTestsBuffers2_t>* clone = classEstimatorFinalizer->Clone();
    delete classEstimatorFinalizer;
    VectorBufferTemplate<double> final_estimator = clone->Finalize(estimator.Sum(), estimator);

    double expected_result_data[] = {0, 0.1, 0.2, 0.3, 0.4};
    VectorBufferTemplate<double> expected_result(&expected_result_data[0], 5);

    BOOST_CHECK( final_estimator == expected_result );
    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()