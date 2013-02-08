#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <ctime>
#include <cstdio>
#include <climits>

#include "assert_util.h"
#include "RandomThresholdHistogramDataCollector.h"


RandomThresholdHistogramDataCollector::RandomThresholdHistogramDataCollector(int numberOfClasses,
                                                                            int numberOfThresholds,
                                                                            float probabilityOfNullStream )
: mNumberOfThresholdSamples(0)
, mNumberOfCollectedSamples(0)
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

    UpdateThresholds(featureValues);

    if ( mData.HasFloat32MatrixBuffer(THRESHOLDS) )
    {
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

        const Int32VectorBuffer& classLabels = data.GetInt32VectorBuffer(CLASS_LABELS).Slice(sampleIndices);
        const Float32VectorBuffer& sampleWeights = data.GetFloat32VectorBuffer(SAMPLE_WEIGHTS).Slice(sampleIndices);

        Float32Tensor3Buffer& histogramLeft = mData.GetFloat32Tensor3Buffer(HISTOGRAM_LEFT);
        Float32Tensor3Buffer& histogramRight = mData.GetFloat32Tensor3Buffer(HISTOGRAM_RIGHT);
        Float32MatrixBuffer& thresholds = mData.GetFloat32MatrixBuffer(THRESHOLDS);

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
                for(int t=0; t<mNumberOfThresholds; t++)
                {
                    const float threshold = thresholds.Get(f,t);
                    const bool isleft = (featureValue >= threshold);
                    Float32Tensor3Buffer& histogram = isleft ? histogramLeft : histogramRight;
                    const float newClassCount = histogram.Get(f, t, classLabel) + weight;
                    histogram.Set(f, t, classLabel, newClassCount);
                    // printf("class=%d count=%0.2f isLeft=%d threshold=%0.2f\n", classLabel, newClassCount, isleft, threshold);
                }
            }
        }

        mNumberOfCollectedSamples += sampleIndices.GetN();
    }
}

void RandomThresholdHistogramDataCollector::UpdateThresholds(const Float32MatrixBuffer& featureValues)
{
    // If the random thresholds have not been set
    if( !mData.HasFloat32MatrixBuffer(THRESHOLDS) )
    {
        const int numberOfFeatures = featureValues.GetN();

        if( mThresholds.GetM() != numberOfFeatures)
        {
            mThresholds.Resize(numberOfFeatures, mNumberOfThresholds);
        }


        for(unsigned int s=0; s<featureValues.GetM() && mNumberOfThresholdSamples<mNumberOfThresholds; s++)
        {
            for(unsigned int f=0; f<featureValues.GetN(); f++)
            {
                const float value = featureValues.Get(s,f);
                mThresholds.Set(f, mNumberOfThresholdSamples, value);
            }
            mNumberOfThresholdSamples++;
        }

        if(mNumberOfThresholdSamples >= mNumberOfThresholds)
        {
            mData.AddFloat32MatrixBuffer(THRESHOLDS, mThresholds);
        }
    }
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


