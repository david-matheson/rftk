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
    TimeLogger timer(extraInfo, "LogSplitInfo");

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
            	typename BufferTypes::ImpurityValue featureImpurity = typename BufferTypes::ImpurityValue(0);
                for(int t=0; t<splitpointCounts.Get(f); t++)
                {
                    featureImpurity = std::max< typename BufferTypes::ImpurityValue >(featureImpurity, impurities.Get(f,t));
                }

                const bool isSelectedFeature = s == bestSelectorBuffer && f == bestFeature;
    	        if(isSelectedFeature)
    	        {
                    WriteValue<int>(extraInfo, "SplitInfo-PerNode-BufferSelectorId", nodeIndex, bestSelectorBuffer);
                    WriteValue<float>(extraInfo, "SplitInfo-PerNode-Impurity", nodeIndex, featureImpurity);
    	        }

                ssb.mFeatureIndexer->LogFeatureInfo(readCollection, depth, f, featureImpurity, isSelectedFeature, extraInfo);

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