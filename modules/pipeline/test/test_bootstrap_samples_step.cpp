#include <boost/test/unit_test.hpp>

#include <vector>

#include "UniqueBufferId.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "BootstrapSamplesStep.h"

BOOST_AUTO_TEST_SUITE( BootstrapSamplesStepTests )

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    const int numberOfDatapoints = 7;
    MatrixBufferTemplate<float> xs(numberOfDatapoints, 10);
    BufferId xs_key("xs");
    BufferCollection collection;
    collection.AddBuffer< MatrixBufferTemplate<float> > (xs_key, xs);
    BufferCollectionStack stack;
    stack.Push(&collection);

    BootstrapSamplesStep<float,int> bootstrap_step(xs_key);
    bootstrap_step.ProcessStep(stack, collection);

    BOOST_CHECK(collection.HasBuffer< VectorBufferTemplate<float> >(bootstrap_step.WeightsBufferId));
    BOOST_CHECK(collection.HasBuffer< VectorBufferTemplate<int> >(bootstrap_step.IndicesBufferId));

    VectorBufferTemplate<float>& weights =
              collection.GetBuffer< VectorBufferTemplate<float> >(bootstrap_step.WeightsBufferId);

    BOOST_CHECK_CLOSE(weights.Sum(), static_cast<float>(numberOfDatapoints), 0.1);

    VectorBufferTemplate<int>& indices =
              collection.GetBuffer< VectorBufferTemplate<int> >(bootstrap_step.IndicesBufferId);

    float totalWeight = 0.0f;
    std::vector<bool> isUsed(numberOfDatapoints);
    for(int i=0; i<indices.GetN(); i++)
    {
        const int index = indices.Get(i);
        BOOST_CHECK(!isUsed[index]);
        isUsed[index] = true;
        totalWeight += weights.Get(index);
    }

    BOOST_CHECK_CLOSE(totalWeight, static_cast<float>(numberOfDatapoints), 0.1);
}

BOOST_AUTO_TEST_SUITE_END()