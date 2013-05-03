#include <boost/test/unit_test.hpp>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"
#include "AssignStreamStep.h"


BOOST_AUTO_TEST_SUITE( AssignStreamStepTest )

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    BufferId indices_key = "indices";
    VectorBufferTemplate<int> indices(1000);
    BufferCollection bc;
    bc.AddBuffer(indices_key, indices);
    BufferCollectionStack stack;
    stack.Push(&bc);

    AssignStreamStep<float, int> assignStreamStep(indices_key, 0.25);
    boost::mt19937 gen;
    assignStreamStep.ProcessStep(stack, bc, gen);

    VectorBufferTemplate<int>& streamType = bc.GetBuffer< VectorBufferTemplate<int> >(assignStreamStep.StreamTypeBufferId);

    int counts[] = {0, 0};
    for(int i=0; i<indices.GetN(); i++)
    {
        int stream = streamType.Get(i);
        counts[stream]++;
    }

    BOOST_CHECK_CLOSE(float(counts[STREAM_ESTIMATION]), 750, 5);
    BOOST_CHECK_CLOSE(float(counts[STREAM_IMPURITY]), 250, 5);
}

BOOST_AUTO_TEST_SUITE_END()