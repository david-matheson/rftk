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
template <class FloatType, class IntType>
class AssignStreamStep : public PipelineStepI
{
public:
    AssignStreamStep(const BufferId& indicesBufferId,
                     FloatType probabiltyOfImpurityStream );

    AssignStreamStep(const BufferId& indicesBufferId,
                     FloatType probabiltyOfImpurityStream,
                     bool iid );

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    const BufferId StreamTypeBufferId;
private:
    const BufferId mIndicesBufferId;
    const FloatType mProbabilityOfImpurityStream;
    const bool mIid;
};


template <class FloatType, class IntType>
AssignStreamStep<FloatType, IntType>::AssignStreamStep(const BufferId& indicesBufferId,
                                                        FloatType probabiltyOfImpurityStream )
: StreamTypeBufferId(GetBufferId("StreamType"))
, mIndicesBufferId(indicesBufferId)
, mProbabilityOfImpurityStream(probabiltyOfImpurityStream)
, mIid(true)
{}

template <class FloatType, class IntType>
AssignStreamStep<FloatType, IntType>::AssignStreamStep(const BufferId& indicesBufferId,
                                                        FloatType probabiltyOfImpurityStream,
                                                        bool iid )
: StreamTypeBufferId(GetBufferId("StreamType"))
, mIndicesBufferId(indicesBufferId)
, mProbabilityOfImpurityStream(probabiltyOfImpurityStream)
, mIid(iid)
{}


template <class FloatType, class IntType>
PipelineStepI* AssignStreamStep<FloatType, IntType>::Clone() const
{
    AssignStreamStep<FloatType, IntType>* clone = new AssignStreamStep<FloatType, IntType>(*this);
    return clone;
}

template <class FloatType, class IntType>
void AssignStreamStep<FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                        BufferCollection& writeCollection,
                                        boost::mt19937& gen) const
{
    const VectorBufferTemplate<IntType>& indices =
          readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId);

    VectorBufferTemplate<IntType>& streamType =
            writeCollection.GetOrAddBuffer< VectorBufferTemplate<IntType> >(StreamTypeBufferId);
    streamType.Resize( indices.GetN() );

    if( mIid )
    {
        boost::bernoulli_distribution<> impuritystream_bernoulli(mProbabilityOfImpurityStream);
        boost::variate_generator<boost::mt19937&,boost::bernoulli_distribution<> > var_impuritystream_bernoulli(gen, impuritystream_bernoulli);

        for(IntType i=0; i<indices.GetN(); i++)
        {
            streamType.Set( i, var_impuritystream_bernoulli());
        }      
    }
    else
    {
        // Sample without replacement so a dimension is not choosen multiple times
        std::vector<int> streamTypeVec(indices.GetN());
        sampleWithOutReplacement(&streamTypeVec[0], streamTypeVec.size(), 
                                      static_cast<IntType>(static_cast<FloatType>(indices.GetN())*mProbabilityOfImpurityStream));

        for(IntType i=0; i<indices.GetN(); i++)
        {
            streamType.Set( i, streamTypeVec[i]);
        } 
    }
}
