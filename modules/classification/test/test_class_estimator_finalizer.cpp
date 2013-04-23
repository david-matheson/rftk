#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "ClassEstimatorFinalizer.h"

BOOST_AUTO_TEST_SUITE( ClassEstimatorFinalizerTests )

BOOST_AUTO_TEST_CASE(test_Finalize)
{
    double estimator_data[] = {0, 1.0, 2.0, 3.0, 4.0};
    VectorBufferTemplate<double> estimator(&estimator_data[0], 5);

    ClassEstimatorFinalizer<double> classEstimatorFinalizer;
    classEstimatorFinalizer.Finalize(estimator.Sum(), estimator);

    double expected_result_data[] = {0, 0.1, 0.2, 0.3, 0.4};
    VectorBufferTemplate<double> expected_result(&expected_result_data[0], 5);

    BOOST_CHECK( estimator == expected_result );
}

BOOST_AUTO_TEST_CASE(test_Clone)
{
    double estimator_data[] = {0, 1.0, 2.0, 3.0, 4.0};
    VectorBufferTemplate<double> estimator(&estimator_data[0], 5);

    FinalizerI<double>* classEstimatorFinalizer = new ClassEstimatorFinalizer<double>();
    FinalizerI<double>* clone = classEstimatorFinalizer->Clone();
    delete classEstimatorFinalizer;
    clone->Finalize(estimator.Sum(), estimator);

    double expected_result_data[] = {0, 0.1, 0.2, 0.3, 0.4};
    VectorBufferTemplate<double> expected_result(&expected_result_data[0], 5);

    BOOST_CHECK( estimator == expected_result );
    delete clone;
}

BOOST_AUTO_TEST_SUITE_END()