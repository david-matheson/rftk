#include <boost/test/unit_test.hpp>

#include "UniqueBufferId.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "AllSamplesStep.h"

BOOST_AUTO_TEST_SUITE( AllSamplesStepTests )

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    const int numberOfDatapoints = 7;
    MatrixBufferTemplate<float> xs(numberOfDatapoints, 10);
    BufferId xs_key("xs");
    BufferCollection collection;
    collection.AddBuffer< MatrixBufferTemplate<float> > (xs_key, xs);
    BufferCollectionStack stack;
    stack.Push(&collection);

    AllSamplesStep<float,int> all_samples_step(xs_key);
    all_samples_step.ProcessStep(stack, collection);

    BOOST_CHECK(collection.HasBuffer< VectorBufferTemplate<float> >(all_samples_step.WeightsBufferId));
    BOOST_CHECK(collection.HasBuffer< VectorBufferTemplate<int> >(all_samples_step.IndicesBufferId));

    VectorBufferTemplate<float>& weights =
              collection.GetBuffer< VectorBufferTemplate<float> >(all_samples_step.WeightsBufferId);

    VectorBufferTemplate<int>& indices =
              collection.GetBuffer< VectorBufferTemplate<int> >(all_samples_step.IndicesBufferId);

    BOOST_CHECK_CLOSE(weights.Sum(), static_cast<float>(numberOfDatapoints), 0.1);
    BOOST_CHECK_EQUAL(weights.GetN(), numberOfDatapoints);
    BOOST_CHECK_EQUAL(indices.GetN(), numberOfDatapoints);

    for(int i=0; i<numberOfDatapoints; i++)
    {
        BOOST_CHECK_CLOSE(weights.Get(i), 1.0f, 0.1);
        BOOST_CHECK_EQUAL(indices.Get(i), i);
    }
}

BOOST_AUTO_TEST_SUITE_END()