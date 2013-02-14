#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <ctime>
#include <cstdio>
#include <limits>

#include "assert_util.h"
#include "RandomThresholdHistogramDataCollector.h"


RandomThresholdHistogramDataCollector::RandomThresholdHistogramDataCollector(int numberOfClasses,
                                                                            int numberOfThresholds,
                                                                            float probabilityOfNullStream )
: mNumberOfCollectedSamples(0)
, mNumberOfClasses(numberOfClasses)
, mNumberOfThresholds(numberOfThresholds)
, mProbabilityOfNullStream(probabilityOfNullStream)
{
}

RandomThresholdHistogramDataCollector::~RandomThresholdHistogramDataCollector()
{
}

void RandomThresholdHistogramDataCollector::Collect( const BufferCollection& data,
                                    const Int32VectorBuffer& sampleIndices,
                                    const Float32MatrixBuffer& featureValues,
                                    boost::mt19937& gen )
{
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), featureValues.GetM())
    ASSERT(data.HasInt32VectorBuffer(CLASS_LABELS))

    boost::bernoulli_distribution<> nullstream_bernoulli(mProbabilityOfNullStream);
    boost::variate_generator<boost::mt19937&,boost::bernoulli_distribution<> > var_nullstream_bernoulli(gen, nullstream_bernoulli);

    const int numberOfFeatures = featureValues.GetN();
    if( !mData.HasFloat32Tensor3Buffer(HISTOGRAM_LEFT) )
    {
        mData.AddFloat32Tensor3Buffer(HISTOGRAM_LEFT,
          Float32Tensor3Buffer(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses));
    }

    if( !mData.HasFloat32Tensor3Buffer(HISTOGRAM_RIGHT) )
    {
        mData.AddFloat32Tensor3Buffer(HISTOGRAM_RIGHT,
          Float32Tensor3Buffer(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses));
    }

    if ( !mData.HasFloat32MatrixBuffer(THRESHOLDS) )
    {
        mData.AddFloat32MatrixBuffer(THRESHOLDS, 
          Float32MatrixBuffer(numberOfFeatures, mNumberOfThresholds));
    }

    if ( !mData.HasInt32VectorBuffer(THRESHOLD_COUNTS) )
    {
        mData.AddInt32VectorBuffer(THRESHOLD_COUNTS, 
          Int32VectorBuffer(numberOfFeatures));
    }


    const Int32VectorBuffer& classLabels = data.GetInt32VectorBuffer(CLASS_LABELS).Slice(sampleIndices);
    const Float32VectorBuffer& sampleWeights = data.GetFloat32VectorBuffer(SAMPLE_WEIGHTS).Slice(sampleIndices);

    Float32Tensor3Buffer& histogramLeft = mData.GetFloat32Tensor3Buffer(HISTOGRAM_LEFT);
    Float32Tensor3Buffer& histogramRight = mData.GetFloat32Tensor3Buffer(HISTOGRAM_RIGHT);
    Float32MatrixBuffer& thresholds = mData.GetFloat32MatrixBuffer(THRESHOLDS);
    Int32VectorBuffer& thresholdCounts = mData.GetInt32VectorBuffer(THRESHOLD_COUNTS);

    for(int i=0; i<sampleIndices.GetN(); i++)
    {
        if(var_nullstream_bernoulli() > 0)
        {
            continue;
        }

        const int classLabel = classLabels.Get(i);
        const float weight = sampleWeights.Get(i);
        for(int f=0; f<featureValues.GetN(); f++)
        {
            const float featureValue = featureValues.Get(i,f);

            const bool thresholdUpdated = AddThreshold(thresholds, thresholdCounts, f, featureValue);
            if( thresholdUpdated )
            {
                continue;
            }

            for(int t=0; t<thresholdCounts.Get(f); t++)
            {
                const float threshold = thresholds.Get(f,t);
                const bool isleft = (featureValue >= threshold);
                if( isleft )
                {
                    histogramLeft.Incr(f, t, classLabel, weight);
                }
                else
                {
                    histogramRight.Incr(f, t, classLabel, weight);
                }
            }

        }
    }

    mNumberOfCollectedSamples += sampleIndices.GetN();
}

bool RandomThresholdHistogramDataCollector::AddThreshold(Float32MatrixBuffer& thresholds, Int32VectorBuffer& thresholdCounts, 
                                                        const int featureIndex, const float featureValue)
{
    const int currentNumberOfThresholds = thresholdCounts.Get(featureIndex);
    if( currentNumberOfThresholds >= mNumberOfThresholds )
    {
        return false;
    }

    for(int thresholdIndex=0; thresholdIndex<thresholdCounts.Get(featureIndex); thresholdIndex++)
    {
        if(abs(featureValue - thresholds.Get(featureIndex, thresholdIndex)) < std::numeric_limits<float>::epsilon())
        {
            return false;
        }
    }

    thresholds.Set(featureIndex, currentNumberOfThresholds, featureValue);
    thresholdCounts.Incr(featureIndex, 1);
    return true;
}

const BufferCollection& RandomThresholdHistogramDataCollector::GetCollectedData()
{
    return mData;
}

int RandomThresholdHistogramDataCollector::GetNumberOfCollectedSamples()
{
    return mNumberOfCollectedSamples;
}

RandomThresholdHistogramDataCollectorFactory::RandomThresholdHistogramDataCollectorFactory(int numberOfClasses,
                                                                                            int numberOfThresholds,
                                                                                            float probabilityOfNullStream)
: mNumberOfClasses(numberOfClasses)
, mNumberOfThresholds(numberOfThresholds)
, mProbabilityOfNullStream(probabilityOfNullStream)
{
}

RandomThresholdHistogramDataCollectorFactory::~RandomThresholdHistogramDataCollectorFactory()
{}

NodeDataCollectorFactoryI* RandomThresholdHistogramDataCollectorFactory::Clone() const
{
    return new RandomThresholdHistogramDataCollectorFactory(*this);
}

NodeDataCollectorI* RandomThresholdHistogramDataCollectorFactory::Create() const
{
    return new RandomThresholdHistogramDataCollector(mNumberOfClasses, mNumberOfThresholds, mProbabilityOfNullStream);
}


