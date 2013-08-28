#pragma once

#include "unused.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "BufferCollectionUtils.h"


#include "SplitSelectorBuffers.h"
#include "ShouldSplitCriteriaI.h"



template <class BufferTypes>
class LogSplitInfo
{
public:
	LogSplitInfo();
    virtual ~LogSplitInfo() {}

    virtual void Log(const std::vector<SplitSelectorBuffers>& splitSelectorBuffers,
    				const ShouldSplitCriteriaI* shouldSplitCriteria,
    				const BufferCollectionStack& readCollection, int depth,
                    int bestSelectorBuffer, int bestFeature, int bestSplitpoint,
                    BufferCollection& extraInfo, int nodeIndex) const;

    virtual LogSplitInfo* Clone() const;


};

template <class BufferTypes>
LogSplitInfo<BufferTypes>::LogSplitInfo()
{
}


template <class BufferTypes>
void LogSplitInfo<BufferTypes>::Log( const std::vector<SplitSelectorBuffers>& splitSelectorBuffers,
				    				const ShouldSplitCriteriaI* shouldSplitCriteria,
				    				const BufferCollectionStack& readCollection, int depth,
				                    int bestSelectorBuffer, int bestFeature, int bestSplitpoint,
				                    BufferCollection& extraInfo, int nodeIndex) const
{
	UNUSED_PARAM(shouldSplitCriteria)
	UNUSED_PARAM(depth)
	UNUSED_PARAM(bestSplitpoint)

	const int numberOfWaitForBestmSplitSelectorBuffers = splitSelectorBuffers.size();
    for(int s=0; s<numberOfWaitForBestmSplitSelectorBuffers; s++)
    {
        const SplitSelectorBuffers& ssb = splitSelectorBuffers[s];

        if(ssb.mFeatureIndexer != NULL)
        {

            const MatrixBufferTemplate<typename BufferTypes::ImpurityValue>& impurities
               = readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ImpurityValue> >(ssb.mImpurityBufferId);

            const VectorBufferTemplate<typename BufferTypes::Index>& splitpointCounts
                   = readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(ssb.mSplitpointsCountsBufferId);


    	    const int numberOfFeatures = impurities.GetM();
            for(int f=0; f<numberOfFeatures; f++)
            {
            	typename BufferTypes::ImpurityValue featureImpurity = -std::numeric_limits<typename BufferTypes::ImpurityValue>::max();
                for(int t=0; t<splitpointCounts.Get(f); t++)
                {
                    featureImpurity = std::max< typename BufferTypes::ImpurityValue >(featureImpurity, impurities.Get(f,t));
                }

                int featureIndex = ssb.mFeatureIndexer->IndexFeature(readCollection, f);
      
                IncrementValue<int>(extraInfo, "SplitInfo-FeaturesSampled", featureIndex, 1);
                IncrementValue<float>(extraInfo, "SplitInfo-ImpuritySumSampled", featureIndex, featureImpurity);

    	        if(s == bestSelectorBuffer && f == bestFeature)
    	        {
    	        	IncrementValue<int>(extraInfo, "SplitInfo-FeaturesSelected", featureIndex, 1);
    	        	IncrementValue<float>(extraInfo, "SplitInfo-ImpuritySumSelected", featureIndex, featureImpurity);

    	        	WriteValue<float>(extraInfo, "SplitInfo-PerNode-Impurity", nodeIndex, featureImpurity);
                    WriteValue<int>(extraInfo, "SplitInfo-PerNode-BufferSelectorId", nodeIndex, bestSelectorBuffer);
                    WriteValue<int>(extraInfo, "SplitInfo-PerNode-FeatureId", nodeIndex, featureIndex);
    	        }
            }
        }
    }

}

template <class BufferTypes>
LogSplitInfo<BufferTypes>* LogSplitInfo<BufferTypes>::Clone() const
{
    LogSplitInfo<BufferTypes>* clone = new LogSplitInfo<BufferTypes>();
    return clone;
}