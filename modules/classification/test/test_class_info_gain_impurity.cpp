#include <boost/test/unit_test.hpp>

#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "ClassInfoGainImpurity.h"
#include "SplitpointsImpurity.h"


struct ClassInfoGainFixture {
    ClassInfoGainFixture()
    : splitpoint_counts_key("splitpoint_counts")
    , child_counts_key("child_counts")
    , left_histogram_key("left_histogram")
    , right_histogram_key("right_histogram")
    , collection()
    , stack()
    {

        int splitpoint_count_data[] = {3, 1};
        const VectorBufferTemplate<int> splitpoint_counts = VectorBufferTemplate<int>(&splitpoint_count_data[0], 2);
        collection.AddBuffer(splitpoint_counts_key, splitpoint_counts);

        float child_counts_data[] = {15,9, 6,15, 4,12,
                                    9,9, 100,100, 0,0};
        const Tensor3BufferTemplate<float> child_counts = Tensor3BufferTemplate<float>(&child_counts_data[0], 2,3,2);
        collection.AddBuffer(child_counts_key, child_counts);

        float left_histogram_data[] = {10,0,5,  2,2,2, 1,0,3,
                                    9,0,0, 100,0,0, 0,0,0};
        const Tensor3BufferTemplate<float> left_histogram = Tensor3BufferTemplate<float>(&left_histogram_data[0], 2,3,3);
        collection.AddBuffer(left_histogram_key, left_histogram);

        float right_histogram_data[] = {0,9,0,  5,5,5, 5,8,1,
                                    0,0,9, 0,0,100, 0,0,0};
        const Tensor3BufferTemplate<float> right_histogram = Tensor3BufferTemplate<float>(&right_histogram_data[0], 2,3,3);
        collection.AddBuffer(right_histogram_key, right_histogram);

        stack.Push(&collection);
    }

    ~ClassInfoGainFixture()
    {
    }

    const BufferCollectionKey_t splitpoint_counts_key;
    const BufferCollectionKey_t child_counts_key;
    const BufferCollectionKey_t left_histogram_key;
    const BufferCollectionKey_t right_histogram_key;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( ClassInfoGainTests,  ClassInfoGainFixture )

BOOST_AUTO_TEST_CASE(test_ClassInfoGain_Impurity)
{
    const VectorBufferTemplate<int>& splitpoint_counts = stack.GetBuffer< VectorBufferTemplate<int> >(splitpoint_counts_key);
    const Tensor3BufferTemplate<float>& child_counts = stack.GetBuffer< Tensor3BufferTemplate<float> >(child_counts_key);
    const Tensor3BufferTemplate<float>& left_histogram = stack.GetBuffer< Tensor3BufferTemplate<float> >(left_histogram_key);
    const Tensor3BufferTemplate<float>& right_histogram = stack.GetBuffer< Tensor3BufferTemplate<float> >(right_histogram_key);

    ClassInfoGainImpurity< SinglePrecisionBufferTypes > ig;

    BOOST_CHECK_CLOSE(ig.Impurity(0,0, child_counts, left_histogram, right_histogram), 0.954434, 0.1);
    BOOST_CHECK_CLOSE(ig.Impurity(0,1, child_counts, left_histogram, right_histogram), 0.0, 0.1);
    BOOST_CHECK_CLOSE(ig.Impurity(0,2, child_counts, left_histogram, right_histogram), 0.416580, 0.1);
    BOOST_CHECK_CLOSE(ig.Impurity(1,0, child_counts, left_histogram, right_histogram), 1.0, 0.1);
}

BOOST_AUTO_TEST_CASE(test_ClassInfoGain_SplitpointsImpurity_ProcessStep)
{
    SplitpointsImpurity<ClassInfoGainImpurity< SinglePrecisionBufferTypes > > splitpointsImpurity(splitpoint_counts_key,
                                                                                child_counts_key,
                                                                                left_histogram_key,
                                                                                right_histogram_key );
    boost::mt19937 gen(0);
    splitpointsImpurity.ProcessStep(stack, collection, gen, collection, 0);
    const MatrixBufferTemplate<float>& impurities =
            collection.GetBuffer< MatrixBufferTemplate<float> >(splitpointsImpurity.ImpurityBufferId);

    BOOST_CHECK_CLOSE(impurities.Get(0,0), 0.954434, 0.1);
    BOOST_CHECK_CLOSE(impurities.Get(0,1), 0.0, 0.1);
    BOOST_CHECK_CLOSE(impurities.Get(0,2), 0.416580, 0.1);
    BOOST_CHECK_CLOSE(impurities.Get(1,0), 1.0, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()