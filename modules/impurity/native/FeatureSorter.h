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
template <class FloatType>
class FeatureSorter
{
public:
    FeatureSorter(  const MatrixBufferTemplate<FloatType>& featureValues,
                    const FeatureValueOrdering ordering,
                    const int featureIndex );
    void Sort();
    int GetUnSortedIndex(int sortedIndex) const;
    FloatType GetFeatureValue(int sortedIndex) const;
    int GetNumberOfSamples() const;

private:
    const int mNumberOfSamples;
    std::vector< std::pair<FloatType, int> > mValueIndices;
};

template <class FloatType>
FeatureSorter<FloatType>::FeatureSorter(   const MatrixBufferTemplate<FloatType>& featureValues,
                                            const FeatureValueOrdering ordering,
                                            const int featureIndex)
: mNumberOfSamples( ordering == FEATURES_BY_DATAPOINTS ? featureValues.GetN() : featureValues.GetM())
, mValueIndices(mNumberOfSamples)
{
    for(int s=0; s<mNumberOfSamples; s++)
    {
        int r = (ordering == FEATURES_BY_DATAPOINTS) ? featureIndex : s;
        int c = (ordering == FEATURES_BY_DATAPOINTS) ? s : featureIndex;
        mValueIndices[s] = std::pair<FloatType,int>(featureValues.Get(r,c), s);
    }
}

template <class FloatType>
void FeatureSorter<FloatType>::Sort()
{
    std::sort( mValueIndices.begin(), mValueIndices.end() );
}

template <class FloatType>
int FeatureSorter<FloatType>::GetUnSortedIndex(int sortedIndex) const
{
    return mValueIndices[sortedIndex].second;
}

template <class FloatType>
FloatType FeatureSorter<FloatType>::GetFeatureValue(int sortedIndex) const
{
    return mValueIndices[sortedIndex].first;
}

template <class FloatType>
int FeatureSorter<FloatType>::GetNumberOfSamples() const
{
    return mNumberOfSamples;
}