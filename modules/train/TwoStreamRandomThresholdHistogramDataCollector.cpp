#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <ctime>
#include <cstdio>
#include <climits>

#include "asserts/asserts.h"
#include "TwoStreamRandomThresholdHistogramDataCollector.h"


TwoStreamRandomThresholdHistogramDataCollector::TwoStreamRandomThresholdHistogramDataCollector( int numberOfClasses,
                                                                                                int numberOfThresholds,
                                                                                                float probabilityOfNullStream,
                                                                                                float probabilityOfImpurityStream)
: RandomThresholdHistogramDataCollector(numberOfClasses, numberOfThresholds, probabilityOfNullStream)
, mProbabilityOfImpurityStream(probabilityOfImpurityStream)
{
}

TwoStreamRandomThresholdHistogramDataCollector::~TwoStreamRandomThresholdHistogramDataCollector()
{
}

void TwoStreamRandomThresholdHistogramDataCollector::Collect( const BufferCollection& data,
                                    const Int32VectorBuffer& sampleIndices,
                                    const Float32MatrixBuffer& featureValues,
                                    boost::mt19937& gen )
{
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), featureValues.GetM())
    ASSERT(data.HasInt32VectorBuffer(CLASS_LABELS))

    boost::bernoulli_distribution<> nullstream_bernoulli(mProbabilityOfNullStream);
    boost::variate_generator<boost::mt19937&,boost::bernoulli_distribution<> > var_nullstream_bernoulli(gen, nullstream_bernoulli);

    boost::bernoulli_distribution<> impuritystream_bernoulli(mProbabilityOfImpurityStream);
    boost::variate_generator<boost::mt19937&,boost::bernoulli_distribution<> > var_impuritystream_bernoulli(gen, impuritystream_bernoulli);

    const int numberOfFeatures = featureValues.GetN();
    if( !mData.HasFloat32Tensor3Buffer(IMPURITY_HISTOGRAM_LEFT) )
    {
        mData.AddFloat32Tensor3Buffer(IMPURITY_HISTOGRAM_LEFT,
          Float32Tensor3Buffer(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses));
    }

    if( !mData.HasFloat32Tensor3Buffer(IMPURITY_HISTOGRAM_RIGHT) )
    {
        mData.AddFloat32Tensor3Buffer(IMPURITY_HISTOGRAM_RIGHT,
          Float32Tensor3Buffer(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses));
    }

    if( !mData.HasFloat32Tensor3Buffer(YS_HISTOGRAM_LEFT) )
    {
        mData.AddFloat32Tensor3Buffer(YS_HISTOGRAM_LEFT,
          Float32Tensor3Buffer(numberOfFeatures, mNumberOfThresholds, mNumberOfClasses));
    }

    if( !mData.HasFloat32Tensor3Buffer(YS_HISTOGRAM_RIGHT) )
    {
        mData.AddFloat32Tensor3Buffer(YS_HISTOGRAM_RIGHT,
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

    Float32Tensor3Buffer& impurityHistogramLeft = mData.GetFloat32Tensor3Buffer(IMPURITY_HISTOGRAM_LEFT);
    Float32Tensor3Buffer& impurityHistogramRight = mData.GetFloat32Tensor3Buffer(IMPURITY_HISTOGRAM_RIGHT);
    Float32Tensor3Buffer& ysHistogramLeft = mData.GetFloat32Tensor3Buffer(YS_HISTOGRAM_LEFT);
    Float32Tensor3Buffer& ysHistogramRight = mData.GetFloat32Tensor3Buffer(YS_HISTOGRAM_RIGHT);
    Float32MatrixBuffer& thresholds = mData.GetFloat32MatrixBuffer(THRESHOLDS);
    Int32VectorBuffer& thresholdCounts = mData.GetInt32VectorBuffer(THRESHOLD_COUNTS);

    for(int i=0; i<sampleIndices.GetN(); i++)
    {
        if(var_nullstream_bernoulli() > 0)
        {
            continue;
        }

        const bool structureStream = (var_impuritystream_bernoulli() > 0);

        const int classLabel = classLabels.Get(i);
        const float weight = sampleWeights.Get(i);
        for(int f=0; f<featureValues.GetN(); f++)
        {
            const float featureValue = featureValues.Get(i,f);
            if( structureStream )
            {
                const bool thresholdUpdated = AddThreshold(thresholds, thresholdCounts, f, featureValue);
                if( thresholdUpdated )
                {
                    continue;
                }
            }
            for(int t=0; t<thresholdCounts.Get(f); t++)
            {
                const float threshold = thresholds.Get(f,t);
                const bool isleft = (featureValue >= threshold);

                if( structureStream )
                {
                    if( isleft )
                    {
                        impurityHistogramLeft.Incr(f, t, classLabel, weight);
                    }
                    else
                    {
                        impurityHistogramRight.Incr(f, t, classLabel, weight);
                    }
                }
                else
                {
                    if( isleft )
                    {
                        ysHistogramLeft.Incr(f, t, classLabel, weight);
                    }
                    else
                    {
                        ysHistogramRight.Incr(f, t, classLabel, weight);
                    }
                }
            }
        }
    }

    mNumberOfCollectedSamples += sampleIndices.GetN();

}


const BufferCollection& TwoStreamRandomThresholdHistogramDataCollector::GetCollectedData()
{
    return mData;
}

int TwoStreamRandomThresholdHistogramDataCollector::GetNumberOfCollectedSamples()
{
    return mNumberOfCollectedSamples;
}

TwoStreamRandomThresholdHistogramDataCollectorFactory::TwoStreamRandomThresholdHistogramDataCollectorFactory( int numberOfClasses,
                                                                                            int numberOfThresholds,
                                                                                            float probabilityOfNullStream,
                                                                                            float probabilityOfImpurityStream)
: mNumberOfClasses(numberOfClasses)
, mNumberOfThresholds(numberOfThresholds)
, mProbabilityOfNullStream(probabilityOfNullStream)
, mProbabilityOfImpurityStream(probabilityOfImpurityStream)
{
}

TwoStreamRandomThresholdHistogramDataCollectorFactory::~TwoStreamRandomThresholdHistogramDataCollectorFactory()
{}

NodeDataCollectorFactoryI* TwoStreamRandomThresholdHistogramDataCollectorFactory::Clone() const
{
    return new TwoStreamRandomThresholdHistogramDataCollectorFactory(*this);
}

NodeDataCollectorI* TwoStreamRandomThresholdHistogramDataCollectorFactory::Create() const
{
    return new TwoStreamRandomThresholdHistogramDataCollector(mNumberOfClasses,
                                                              mNumberOfThresholds,
                                                              mProbabilityOfNullStream,
                                                              mProbabilityOfImpurityStream);
}


