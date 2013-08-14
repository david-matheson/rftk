#include <boost/test/unit_test.hpp>

#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "SliceBufferStep.h"

template<typename T>
MatrixBufferTemplate<T> Create4x5ExampleMatrix()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    return MatrixBufferTemplate<T>(&data[0], 4, 5);
}

template<typename T>
VectorBufferTemplate<T> CreateExampleVector7()
{
    T data[] = {0, 5, 3, 11, 9, 22, 1};
    return VectorBufferTemplate<T>(&data[0], 7);
}

template<typename T>
VectorBufferTemplate<T> CreateIndices()
{
    T data[] = {1, 3};
    return VectorBufferTemplate<T>(&data[0], 2);
}

struct SliceBufferFixture {

    SliceBufferFixture()
    : xs_key("xs")
    , ys_key("ys")
    , indices_key("indices")
    , xs(Create4x5ExampleMatrix<double>())
    , ys(CreateExampleVector7<long long>())
    , indices(CreateIndices<int>())
    , collection()
    , stack()
    {
        collection.AddBuffer(xs_key, xs);
        collection.AddBuffer(ys_key, ys);
        collection.AddBuffer(indices_key, indices);
        stack.Push(&collection);
    }

    ~SliceBufferFixture()
    {
    }

    typedef SinglePrecisionBufferTypes BufferTypes_t;

    const BufferCollectionKey_t xs_key;
    const BufferCollectionKey_t ys_key;
    const BufferCollectionKey_t indices_key;
    const MatrixBufferTemplate<double> xs;
    const VectorBufferTemplate<long long> ys;
    const VectorBufferTemplate<int> indices;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( SliceBufferTests,  SliceBufferFixture)

BOOST_AUTO_TEST_CASE(test_SliceBuffer_ProcessStep_matrix_buffer)
{
    const SliceBufferStep< BufferTypes_t, MatrixBufferTemplate<double> > sliceBufferStep(xs_key, indices_key);

    BOOST_CHECK(!collection.HasBuffer< MatrixBufferTemplate<double> >(sliceBufferStep.SlicedBufferId));
    boost::mt19937 gen(0);
    sliceBufferStep.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.HasBuffer< MatrixBufferTemplate<double> >(sliceBufferStep.SlicedBufferId));

    double data[] = {5, 6, 7, 8, 9, 15, 16, 17, 18, 19};
    MatrixBufferTemplate<double> expected_slice_results(&data[0], 2, 5);
    MatrixBufferTemplate<double> slice_results = 
        collection.GetBuffer< MatrixBufferTemplate<double> >(sliceBufferStep.SlicedBufferId);
    BOOST_CHECK(slice_results == expected_slice_results);
}

BOOST_AUTO_TEST_CASE(test_SliceBuffer_ProcessStep_vector_buffer)
{
    const SliceBufferStep< BufferTypes_t, VectorBufferTemplate<long long> > sliceBufferStep(ys_key, indices_key);

    BOOST_CHECK(!collection.HasBuffer< VectorBufferTemplate<long long> >(sliceBufferStep.SlicedBufferId));
    boost::mt19937 gen(0);
    sliceBufferStep.ProcessStep(stack, collection, gen);
    BOOST_CHECK(collection.HasBuffer< VectorBufferTemplate<long long> >(sliceBufferStep.SlicedBufferId));

    long long data[] = {5, 11};
    VectorBufferTemplate<long long> expected_slice_results(&data[0], 2);
    VectorBufferTemplate<long long> slice_results = 
        collection.GetBuffer< VectorBufferTemplate<long long> >(sliceBufferStep.SlicedBufferId);

    BOOST_CHECK(slice_results == expected_slice_results);
}

BOOST_AUTO_TEST_SUITE_END()
