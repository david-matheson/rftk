#include <boost/test/unit_test.hpp>

#include <asserts.h>
#include <BufferCollectionStack.h>
#include <BufferCollection.h>
#include <PipelineStepI.h>
#include <Pipeline.h>

// ----------------------------------------------------------------------------
//
// Mock class that doubles the contents of a VectorBufferTemplate<double>
//
// ----------------------------------------------------------------------------
class DoubleStep: public PipelineStepI
{
public:
    DoubleStep(BufferCollectionKey_t inBufferKey, BufferCollectionKey_t outBufferKey)
    : mInBufferKey(inBufferKey)
    , mOutBufferKey(outBufferKey)
    {}

    ~DoubleStep()
    {}

    void ProcessStep(   const BufferCollectionStack& readCollection,
                        BufferCollection& writeCollection,
                        boost::mt19937& gen,
                        BufferCollection& extraInfo, int nodeIndex) const
    {
        UNUSED_PARAM(extraInfo);
        UNUSED_PARAM(nodeIndex);

        const VectorBufferTemplate<double>& read_buf = readCollection.GetBuffer< VectorBufferTemplate<double> >(mInBufferKey);
        VectorBufferTemplate<double>& write_buf = writeCollection.GetBuffer< VectorBufferTemplate<double> >(mOutBufferKey);
        for(int i=0; i<read_buf.GetN(); i++)
        {
            const double doubleValue = 2.0*read_buf.Get(i);
            write_buf.Set(i, doubleValue);
        }
    }

    virtual PipelineStepI* Clone() const
    {
        DoubleStep* clone = new DoubleStep(*this);
        return clone;
    }

private:
    BufferCollectionKey_t mInBufferKey;
    BufferCollectionKey_t mOutBufferKey;
};


template<typename T>
VectorBufferTemplate<T> CreateExampleVector()
{
    T data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    return VectorBufferTemplate<T>(&data[0], 16);
}

template<typename T>
void CheckMultiple(const VectorBufferTemplate<T>& input, const T multiple, const VectorBufferTemplate<T>& expected)
{
    BOOST_CHECK(input.GetN() == expected.GetN());
    for(int i=0; i<input.GetN(); i++)
    {
        const T value = input.Get(i)*multiple;
        BOOST_CHECK_CLOSE(value, expected.Get(i), 0.001);
    }
}

struct PipelineTestFixture {
    PipelineTestFixture()
    : buffer_name("test_buffer")
    , test_buffer(CreateExampleVector<double>())
    , collection()
    , stack()
    {
        collection.AddBuffer(buffer_name, test_buffer);
        stack.Push(&collection);
    }

    ~PipelineTestFixture()
    {
    }

    const std::string buffer_name;
    const VectorBufferTemplate<double> test_buffer;
    BufferCollection collection;
    BufferCollectionStack stack;
};

BOOST_FIXTURE_TEST_SUITE( PipelineTest, PipelineTestFixture )

BOOST_AUTO_TEST_CASE(test_one_step_pipeline)
{
    VectorBufferTemplate<double>& result_buffer = collection.GetBuffer< VectorBufferTemplate<double> >(buffer_name);
    CheckMultiple<double>(test_buffer, 1.0, result_buffer);

    std::vector<PipelineStepI*> steps;
    DoubleStep double_step(buffer_name, buffer_name);
    steps.push_back(&double_step);
    Pipeline pipeline(steps);

    boost::mt19937 gen(0);
    pipeline.ProcessStep(stack, collection, gen, collection, 0);

    result_buffer = collection.GetBuffer< VectorBufferTemplate<double> >(buffer_name);
    CheckMultiple<double>(test_buffer, 2.0, result_buffer);
}

BOOST_AUTO_TEST_CASE(test_two_step_pipeline)
{
    VectorBufferTemplate<double>& result_buffer = collection.GetBuffer< VectorBufferTemplate<double> >(buffer_name);
    CheckMultiple<double>(test_buffer, 1.0, result_buffer);

    DoubleStep double_step(buffer_name, buffer_name);

    std::vector<PipelineStepI*> steps;
    steps.push_back(&double_step);
    steps.push_back(&double_step);
    Pipeline pipeline(steps);

    boost::mt19937 gen(0);
    pipeline.ProcessStep(stack, collection, gen, collection, 0);

    result_buffer = collection.GetBuffer< VectorBufferTemplate<double> >(buffer_name);
    CheckMultiple<double>(test_buffer, 4.0, result_buffer);
}

BOOST_AUTO_TEST_CASE(test_pipeline_of_pipeline)
{
    VectorBufferTemplate<double>& result_buffer = collection.GetBuffer< VectorBufferTemplate<double> >(buffer_name);
    CheckMultiple<double>(test_buffer, 1.0, result_buffer);

    DoubleStep double_step(buffer_name, buffer_name);

    std::vector<PipelineStepI*> inner_steps;
    inner_steps.push_back(&double_step);
    inner_steps.push_back(&double_step);
    Pipeline inner_pipeline(inner_steps);

    std::vector<PipelineStepI*> outer_steps;
    outer_steps.push_back(&inner_pipeline);
    outer_steps.push_back(&inner_pipeline);
    Pipeline outer_pipeline(outer_steps);

    boost::mt19937 gen(0);
    outer_pipeline.ProcessStep(stack, collection, gen, collection, 0);

    result_buffer = collection.GetBuffer< VectorBufferTemplate<double> >(buffer_name);
    CheckMultiple<double>(test_buffer, 16.0, result_buffer);
}

BOOST_AUTO_TEST_SUITE_END()
