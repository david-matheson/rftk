#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <ctime>
#include <cstdio>

#include "assert_util.h"
#include "RandomThresholdHistogramDataCollector.h"


RandomThresholdHistogramDataCollector::RandomThresholdHistogramDataCollector(int numberOfClasses,
                                                                            int numberOfThresholds,
                                                                            int numberOfSamplesToEstimateThresholds)
: mNumberOfCollectedSamples(0)
, mNumberOfClasses(numberOfClasses)
, mNumberOfThresholds(numberOfThresholds)
, mNumberOfSamplesToEstimateThresholds(numberOfSamplesToEstimateThresholds)
{
}

RandomThresholdHistogramDataCollector::~RandomThresholdHistogramDataCollector()
{
}

void RandomThresholdHistogramDataCollector::Collect( BufferCollection& data,
                                    const MatrixBufferInt& sampleIndices,
                                    const MatrixBufferFloat& featureValues )
{
    // printf("RandomThresholdHistogramDataCollector::Collect\n");

    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(sampleIndices.GetM(), featureValues.GetM())
    ASSERT(data.HasMatrixBufferInt(CLASS_LABELS))

    const int numberOfFeatures = featureValues.GetN();

    if( mNumberOfSamplesToEstimateThresholds > 0 )
    {
        if( mFeaturesForThresholds.GetN() != numberOfFeatures )
        {
            mFeaturesForThresholds = featureValues;
        }
        else
        {
            mFeaturesForThresholds.AppendVertical(featureValues);
        }
        mNumberOfSamplesToEstimateThresholds -= featureValues.GetM();
    }

    if ( mNumberOfSamplesToEstimateThresholds <= 0 )
    {
        if( !mData.HasMatrixBufferFloat(THRESHOLDS) )
        {
            MatrixBufferFloat thresholds = MatrixBufferFloat(numberOfFeatures, mNumberOfThresholds);
            boost::mt19937 gen( std::time(NULL) );
            for(int f=0; f<featureValues.GetN(); f++)
            {
                const float minFeatureValue = mFeaturesForThresholds.GetMin();//minBuffer.Get(f,0);
                const float maxFeatureValue = mFeaturesForThresholds.GetMax();//maxBuffer.Get(f,0);

                boost::uniform_real<float> uniform(minFeatureValue, maxFeatureValue);
                for(int t=0; t<mNumberOfThresholds; t++)
                {
                    const float randomThreshold = uniform(gen);
                    thresholds.Set(f,t, randomThreshold);
                }

                mData.AddMatrixBufferFloat(THRESHOLDS, thresholds);
            }
        }

        if( !mData.HasImgBufferFloat(HISTOGRAM_LEFT) )
        {
            ImgBufferFloat histogramLeft = ImgBufferFloat(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses);
            mData.AddImgBufferFloat(HISTOGRAM_LEFT, histogramLeft);
        }

        if( !mData.HasImgBufferFloat(HISTOGRAM_RIGHT) )
        {
            ImgBufferFloat histogramRight = ImgBufferFloat(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses);
            mData.AddImgBufferFloat(HISTOGRAM_RIGHT, histogramRight);
        }

        MatrixBufferInt classLabels = data.GetMatrixBufferInt(CLASS_LABELS).Slice(sampleIndices);
        MatrixBufferFloat sampleWeights = data.GetMatrixBufferFloat(SAMPLE_WEIGHTS).Slice(sampleIndices);

        ImgBufferFloat histogramLeft = mData.GetImgBufferFloat(HISTOGRAM_LEFT);
        ImgBufferFloat histogramRight = mData.GetImgBufferFloat(HISTOGRAM_RIGHT);
        MatrixBufferFloat thresholds = mData.GetMatrixBufferFloat(THRESHOLDS);

        for(int i=0; i<sampleIndices.GetM(); i++)
        {
            const int classLabel = classLabels.Get(i,0);
            const float weight = sampleWeights.Get(i,0);
            for(int f=0; f<featureValues.GetN(); f++)
            {
                const float featureValue = featureValues.Get(i,f);
                for(int t=0; t<mNumberOfThresholds; t++)
                {
                    const float threshold = thresholds.Get(f,t);
                    const bool isleft = (featureValue >= threshold);
                    ImgBufferFloat& histogram = isleft ? histogramLeft : histogramRight;
                    const float newClassCount = histogram.Get(f, t, classLabel) + weight;
                    histogram.Set(f, t, classLabel, newClassCount);
                    // printf("class=%d count=%0.2f isLeft=%d threshold=%0.2f\n", classLabel, newClassCount, isleft, threshold);
                }
            }
        }

        mNumberOfCollectedSamples += sampleIndices.GetM();
    }
}

BufferCollection RandomThresholdHistogramDataCollector::GetCollectedData()
{
    return mData;
}

int RandomThresholdHistogramDataCollector::GetNumberOfCollectedSamples()
{
    return mNumberOfCollectedSamples;
}

RandomThresholdHistogramDataCollectorFactory::RandomThresholdHistogramDataCollectorFactory(int numberOfClasses,
                                                                                            int numberOfThresholds,
                                                                                            int numberOfSamplesToEstimateThresholds)
: mNumberOfClasses(numberOfClasses)
, mNumberOfThresholds(numberOfThresholds)
, mNumberOfSamplesToEstimateThresholds(numberOfSamplesToEstimateThresholds)
{
}

RandomThresholdHistogramDataCollectorFactory::~RandomThresholdHistogramDataCollectorFactory()
{}

NodeDataCollectorI* RandomThresholdHistogramDataCollectorFactory::Create() const
{
    return new RandomThresholdHistogramDataCollector(mNumberOfClasses, mNumberOfThresholds, mNumberOfSamplesToEstimateThresholds);
}


