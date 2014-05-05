#include <boost/test/unit_test.hpp>

#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "BufferTypes.h"
#include "RandomUniformSplitpointsInRangeStep.h"


template<typename T>
MatrixBufferTemplate<T> CreateFeatureValueRangesBuffer()
{
    T data[] = {0.2, 0.6,
                -30, -0.3,
                -5, 5.1};
    return MatrixBufferTemplate<T>(&data[0], 3, 2);
}


struct RandomUniformSplitpointsInRangeStepFixture {
    RandomUniformSplitpointsInRangeStepFixture()
    : featureValueRangesBufferId("featureValueRangesBufferId")
    , featureValueRangesBuffer(CreateFeatureValueRangesBuffer<float>())
    , collection()
    , stack()
    {
        collection.AddBuffer< MatrixBufferTemplate<float> >(featureValueRangesBufferId, featureValueRangesBuffer);
        stack.Push(&collection);
    }

    ~RandomUniformSplitpointsInRangeStepFixture()
    {
    }

    const BufferId featureValueRangesBufferId;
    const MatrixBufferTemplate<float> featureValueRangesBuffer;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( RandomUniformSplitpointsInRangeStepTests, RandomUniformSplitpointsInRangeStepFixture )

BOOST_AUTO_TEST_CASE(test_ProcessStep)
{
    const int numberOfSplitPoints = 12;
    const RandomUniformSplitpointsInRangeStep< SinglePrecisionBufferTypes > uniformSplitpoints(featureValueRangesBufferId, numberOfSplitPoints);
    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<float> >(uniformSplitpoints.SplitpointsBufferId));
    BOOST_CHECK(!collection.HasBuffer< VectorBufferTemplate<int> >(uniformSplitpoints.SplitpointsCountsBufferId));
    boost::mt19937 gen(0);
    uniformSplitpoints.ProcessStep(stack, collection, gen, collection, 0);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<float> >(uniformSplitpoints.SplitpointsBufferId));
    BOOST_CHECK(collection.HasBuffer< VectorBufferTemplate<int> >(uniformSplitpoints.SplitpointsCountsBufferId));
    const MatrixBufferTemplate<float>& splitpoints = collection.GetBuffer< MatrixBufferTemplate<float> >(uniformSplitpoints.SplitpointsBufferId);
    const VectorBufferTemplate<int>& splitpointCounts = collection.GetBuffer< VectorBufferTemplate<int> >(uniformSplitpoints.SplitpointsCountsBufferId);
    BOOST_CHECK_EQUAL(splitpoints.GetM(), 3);
    BOOST_CHECK_EQUAL(splitpoints.GetN(), numberOfSplitPoints);
    BOOST_CHECK_EQUAL(splitpointCounts.GetN(), splitpoints.GetM());

    for(int r=0; r<splitpoints.GetM(); r++)
    {
        BOOST_CHECK_EQUAL(splitpointCounts.Get(r), numberOfSplitPoints);
        float previousSplitpoint = 0.0f;
        for(int s=0; s<numberOfSplitPoints; s++)
        {
            BOOST_CHECK( splitpoints.Get(r,s) > featureValueRangesBuffer.Get(r,0) );
            BOOST_CHECK( splitpoints.Get(r,s) < featureValueRangesBuffer.Get(r,1) );  
            BOOST_CHECK( splitpoints.Get(r,s) != previousSplitpoint ); 
            previousSplitpoint = splitpoints.Get(r,s);    
        } 
    }
}


BOOST_AUTO_TEST_SUITE_END()
