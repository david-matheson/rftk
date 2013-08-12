#pragma once

#include <vector>
#include <utility>
#include <algorithm>

#include "MatrixBuffer.h"
#include "FeatureExtractorStep.h"

// ----------------------------------------------------------------------------
//
// Sort feature values to get a mapping from unsorted indices to sorted 
// indices.  Either by row or column.
//
// ----------------------------------------------------------------------------
template <class FeatureValueType>
class FeatureSorter
{
public:
    FeatureSorter(  const MatrixBufferTemplate<FeatureValueType>& featureValues,
                    const FeatureValueOrdering ordering,
                    const int featureIndex );
    void Sort();
    int GetUnSortedIndex(int sortedIndex) const;
    FeatureValueType GetFeatureValue(int sortedIndex) const;
    int GetNumberOfSamples() const;

private:
    const int mNumberOfSamples;
    std::vector< std::pair<FeatureValueType, int> > mValueIndices;
};

template <class FeatureValueType>
FeatureSorter<FeatureValueType>::FeatureSorter(   const MatrixBufferTemplate<FeatureValueType>& featureValues,
                                            const FeatureValueOrdering ordering,
                                            const int featureIndex)
: mNumberOfSamples( ordering == FEATURES_BY_DATAPOINTS ? featureValues.GetN() : featureValues.GetM())
, mValueIndices(mNumberOfSamples)
{
    for(int s=0; s<mNumberOfSamples; s++)
    {
        int r = (ordering == FEATURES_BY_DATAPOINTS) ? featureIndex : s;
        int c = (ordering == FEATURES_BY_DATAPOINTS) ? s : featureIndex;
        mValueIndices[s] = std::pair<FeatureValueType,int>(featureValues.Get(r,c), s);
    }
}

template <class FeatureValueType>
void FeatureSorter<FeatureValueType>::Sort()
{
    std::sort( mValueIndices.begin(), mValueIndices.end() );
}

template <class FeatureValueType>
int FeatureSorter<FeatureValueType>::GetUnSortedIndex(int sortedIndex) const
{
    return mValueIndices[sortedIndex].second;
}

template <class FeatureValueType>
FeatureValueType FeatureSorter<FeatureValueType>::GetFeatureValue(int sortedIndex) const
{
    return mValueIndices[sortedIndex].first;
}

template <class FeatureValueType>
int FeatureSorter<FeatureValueType>::GetNumberOfSamples() const
{
    return mNumberOfSamples;
}