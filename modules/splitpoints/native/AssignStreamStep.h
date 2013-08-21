#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/bernoulli_distribution.hpp>

#include "bootstrap.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"


enum
{
    STREAM_ESTIMATION = 0,
    STREAM_STRUCTURE = 1,
};

// ----------------------------------------------------------------------------
//
// Assign a stream (0 or 1) to each index
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class AssignStreamStep : public PipelineStepI
{
public:
    AssignStreamStep(const BufferId& weightsBufferId,
                     double probabiltyOfImpurityStream );

    AssignStreamStep(const BufferId& indicesBufferId,
                     double probabiltyOfImpurityStream,
                     bool iid );

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    const BufferId StreamTypeBufferId;
private:
    const BufferId mWeightsBufferId;
    const typename BufferTypes::ParamsContinuous mProbabilityOfImpurityStream;
    const bool mIid;
};


template <class BufferTypes>
AssignStreamStep<BufferTypes>::AssignStreamStep(const BufferId& weightsBufferId,
                                                double probabiltyOfImpurityStream )
: StreamTypeBufferId(GetBufferId("StreamType"))
, mWeightsBufferId(weightsBufferId)
, mProbabilityOfImpurityStream(probabiltyOfImpurityStream)
, mIid(true)
{}

template <class BufferTypes>
AssignStreamStep<BufferTypes>::AssignStreamStep(const BufferId& weightsBufferId,
                                                double probabiltyOfImpurityStream,
                                                bool iid )
: StreamTypeBufferId(GetBufferId("StreamType"))
, mWeightsBufferId(weightsBufferId)
, mProbabilityOfImpurityStream(probabiltyOfImpurityStream)
, mIid(iid)
{}


template <class BufferTypes>
PipelineStepI* AssignStreamStep<BufferTypes>::Clone() const
{
    AssignStreamStep<BufferTypes>* clone = new AssignStreamStep<BufferTypes>(*this);
    return clone;
}

template <class BufferTypes>
void AssignStreamStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                                BufferCollection& writeCollection,
                                                boost::mt19937& gen,
                                                BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    
    const VectorBufferTemplate<typename BufferTypes::ParamsContinuous>& weights =
          readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::ParamsContinuous> >(mWeightsBufferId);

    VectorBufferTemplate<typename BufferTypes::ParamsInteger>& streamType =
            writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::ParamsInteger> >(StreamTypeBufferId);
    const typename BufferTypes::Index numberOfDatapoints = weights.GetN();
    streamType.Resize( numberOfDatapoints );

    if( mIid )
    {
        boost::bernoulli_distribution<> impuritystream_bernoulli(mProbabilityOfImpurityStream);
        boost::variate_generator<boost::mt19937&,boost::bernoulli_distribution<> > var_impuritystream_bernoulli(gen, impuritystream_bernoulli);

        for(typename BufferTypes::Index i=0; i<numberOfDatapoints; i++)
        {
            streamType.Set( i, var_impuritystream_bernoulli());
        }
    }
    else
    {
        // Sample without replacement so a dimension is not choosen multiple times
        std::vector<typename BufferTypes::Index> streamTypeVec(numberOfDatapoints);
        sampleWithOutReplacement(&streamTypeVec[0], streamTypeVec.size(),
                                      static_cast<typename BufferTypes::Index>(
                                            static_cast<typename BufferTypes::ParamsContinuous>(numberOfDatapoints)*mProbabilityOfImpurityStream));

        for(typename BufferTypes::Index i=0; i<numberOfDatapoints; i++)
        {
            streamType.Set( i, streamTypeVec[i]);
        }
    }
}
