#pragma once

#include <string>

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <BufferCollection.h>
#include <BufferCollectionStack.h>

// ----------------------------------------------------------------------------
//
// PipelineStepI reads inputs from readCollection does some processing and
// writes the results to writeCollection
//
// ----------------------------------------------------------------------------

class PipelineStepI
{
public:
    PipelineStepI()
    : mName("UnknownStep")
    {}

	PipelineStepI(const std::string& name)
    : mName(name)
    {}

    virtual ~PipelineStepI()
    {}

    virtual PipelineStepI* Clone() const = 0;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const = 0;

    const std::string& GetName()
    {
        return mName;
    }

private:
	std::string mName;
};

enum SetRule
{
    WHEN_NEW,
    EVERY_PROCESS
};

