#include <math.h>
#include <stdio.h>
#include <stdexcept>
#include "float.h"

#include "../bootstrap/bootstrap.h"

#include "assert_util.h"

#include "Sorter.h"
#include "ClassInfoGainAllThresholdsBestSplit.h"



float calculateDiscreteEntropy(const float* classLabelCounts, const float* logClassLabelCounts, const int numberOfClasses, float totalCounts)
{
    const float inverseTotalCounts = 1.0f / totalCounts;
    const float logTotalCounts = log(totalCounts);

    float entropy = 0.0f;
    for(int i=0; i<numberOfClasses && totalCounts > 0.0f; i++)
    {
        const float prob = inverseTotalCounts * classLabelCounts[i];
        entropy -= prob * (logClassLabelCounts[i] - logTotalCounts);

        // printf("calculateDiscreteEntropy prob=%0.2f entropy=%0.2f classLabelCounts[%d]=%0.2f\n", prob, entropy, i, classLabelCounts[i]);
    }
    // printf("calculateDiscreteEntropy entropy=%0.2f\n", entropy);

    return entropy;
}


ClassInfoGainAllThresholdsBestSplit::ClassInfoGainAllThresholdsBestSplit(   float ratioOfThresholdsToTest,
                                                                            int minNumberThresholdsToTest,
                                                                            int maxClass)
: mRatioOfThresholdsToTest(ratioOfThresholdsToTest)
, mMinNumberThresholdsToTest(minNumberThresholdsToTest)
, mMaxClass(maxClass+1) //+1 is because 0 is also a valid class
{
}

ClassInfoGainAllThresholdsBestSplit::~ClassInfoGainAllThresholdsBestSplit()
{
}


void ClassInfoGainAllThresholdsBestSplit::BestSplits(   BufferCollection& data,
                                                        // const MatrixBufferInt& sampleIndices,
                                                        // const MatrixBufferFloat& featureValues, // contained in data (if needed)
                                                        MatrixBufferFloat& impurityOut,
                                                        MatrixBufferFloat& thresholdOut,
                                                        MatrixBufferFloat& childCountsOut,
                                                        MatrixBufferFloat& leftYsOut,
                                                        MatrixBufferFloat& rightYsOut) const
// void ClassInfoGainAllThresholdsBestSplit::BestSplits(   BufferCollection& data,
//                                                         const MatrixBufferInt& sampleIndices,
//                                                         const MatrixBufferFloat& featureValues,
//                                                         MatrixBufferFloat& impurityOut,
//                                                         MatrixBufferFloat& thresholdOut)
{
    ASSERT( data.HasMatrixBufferInt(CLASS_LABELS) )
    ASSERT( data.HasMatrixBufferFloat(SAMPLE_WEIGHTS) )
    ASSERT( data.HasMatrixBufferFloat(FEATURE_VALUES) )

    const MatrixBufferInt classLabels = data.GetMatrixBufferInt(CLASS_LABELS);
    const MatrixBufferFloat sampleWeights = data.GetMatrixBufferFloat(SAMPLE_WEIGHTS);
    const MatrixBufferFloat featureValues = data.GetMatrixBufferFloat(FEATURE_VALUES);

    ASSERT_ARG_DIM_1D(classLabels.GetN(), 1)
    ASSERT_ARG_DIM_1D(sampleWeights.GetN(), 1)
    ASSERT_ARG_DIM_1D(classLabels.GetM(), sampleWeights.GetM())

    const int numberSampleIndices = featureValues.GetN();
    const int numberOfFeatures = featureValues.GetM();


    // Create new results buffer if they're not the right dimensions
    if( impurityOut.GetM() != numberOfFeatures || impurityOut.GetN() != 1 )
    {
        impurityOut = MatrixBufferFloat(numberOfFeatures, 1);
    }
    if( thresholdOut.GetM() != numberOfFeatures || thresholdOut.GetN() != 1 )
    {
        thresholdOut = MatrixBufferFloat(numberOfFeatures, 1);
    }
    if( childCountsOut.GetM() != numberOfFeatures || childCountsOut.GetN() != 1 )
    {
        childCountsOut = MatrixBufferFloat(numberOfFeatures, 2);
    }
    if( leftYsOut.GetM() != numberOfFeatures || leftYsOut.GetN() != mMaxClass )
    {
        leftYsOut = MatrixBufferFloat(numberOfFeatures, mMaxClass);
    }
    if( rightYsOut.GetM() != numberOfFeatures || rightYsOut.GetN() != mMaxClass )
    {
        rightYsOut = MatrixBufferFloat(numberOfFeatures, mMaxClass);
    }

    // Initial class histogram and total class weights
    std::vector<float> initialClassLabelCounts(mMaxClass);

    float totalWeight = 0.0f;
    for(int i=0; i<numberSampleIndices; i++)
    {
        const float weight = sampleWeights.Get(i, 0);
        unsigned short classId = classLabels.Get(i, 0);
        initialClassLabelCounts[classId] += weight;
        totalWeight += weight;
    }

    // Work in log space
    std::vector<float> initialLogClassLabelCounts(mMaxClass);
    for(int c=0; c<mMaxClass; c++)
    {
        initialLogClassLabelCounts[c] = initialClassLabelCounts[c] > 0.0f ? log(initialClassLabelCounts[c]) : 0.0f;
    }

    // Determine which thresholds to sample
    std::vector<int> thresholdsToTest(numberSampleIndices);
    int numberOfThresholdsToTest = static_cast<int>( mRatioOfThresholdsToTest * static_cast<float>(numberSampleIndices) );
    numberOfThresholdsToTest = numberOfThresholdsToTest > mMinNumberThresholdsToTest ? numberOfThresholdsToTest : mMinNumberThresholdsToTest;
    sampleWithOutReplacement(&thresholdsToTest[0], thresholdsToTest.size(), numberOfThresholdsToTest);

    const float entropyStart = calculateDiscreteEntropy(&initialClassLabelCounts[0], &initialLogClassLabelCounts[0], mMaxClass, totalWeight);

    std::vector<float> leftClassLabelCounts(mMaxClass);
    std::vector<float> rightClassLabelCounts(mMaxClass);

    std::vector<float> leftLogClassLabelCounts(mMaxClass);
    std::vector<float> rightLogClassLabelCounts(mMaxClass);

    std::vector<bool> recomputeClassLog(mMaxClass);

    std::vector<float> bestLeftClassLabelCounts(mMaxClass);
    std::vector<float> bestRightClassLabelCounts(mMaxClass);

    for(int testIndex=0; testIndex<numberOfFeatures; testIndex++)
    {
        float bestGainInEntropy = FLT_MIN;
        float bestThreshold = FLT_MIN;
        float leftWeight = totalWeight;
        float rightWeight = 0.0f;

        float bestLeftWeight = 0.0f;
        float bestRightWeight = 0.0f;

        //Reset class counts
        for(int c=0; c<mMaxClass; c++)
        {
            leftClassLabelCounts[c] = initialClassLabelCounts[c];
            rightClassLabelCounts[c] = 0.0f;

            leftLogClassLabelCounts[c] = initialLogClassLabelCounts[c];
            rightLogClassLabelCounts[c] = 0.0f;

            bestLeftClassLabelCounts[c] = 0.0f;
            bestRightClassLabelCounts[c] = 0.0f;

            recomputeClassLog[c] = false;
        }

        //Sort the feature values
        const float* featureValuesForTest = featureValues.GetRowPtrUnsafe(testIndex);
        Sorter sorter(featureValuesForTest, numberSampleIndices);
        sorter.Sort();

        //Walk sorted list and update entropy
        for(int sortedIndex=0; sortedIndex<numberSampleIndices-1; sortedIndex++)
        {
            const int i = sorter.GetUnSortedIndex(sortedIndex);
            const float weight = sampleWeights.Get(i, 0);
            const int classId = classLabels.Get(i, 0);

            leftClassLabelCounts[classId] -= weight;
            rightClassLabelCounts[classId] += weight;
            recomputeClassLog[classId] = true;

            leftWeight -= weight;
            rightWeight += weight;

            // Test this threshold
            if( thresholdsToTest[sortedIndex] > 0 )
            {
                for(int c=0; c<mMaxClass; c++)
                {
                    // Recompute the class counts that have changed since the last update
                    if(recomputeClassLog[c])
                    {
                        leftLogClassLabelCounts[c] = leftClassLabelCounts[c] > 0.0f ? log(leftClassLabelCounts[c]) : 0.0f;
                        rightLogClassLabelCounts[c] = rightClassLabelCounts[c] > 0.0f ? log(rightClassLabelCounts[c]) : 0.0f;
                        recomputeClassLog[c] = false;
                    }
                }

                const float leftEntropy = (leftWeight / (leftWeight + rightWeight)) *
                                                calculateDiscreteEntropy(&leftClassLabelCounts[0], &leftLogClassLabelCounts[0], mMaxClass, leftWeight);
                const float rightEntropy = (rightWeight / (leftWeight + rightWeight)) *
                                                calculateDiscreteEntropy(&rightClassLabelCounts[0], &rightLogClassLabelCounts[0], mMaxClass, rightWeight);

                if( (entropyStart - leftEntropy - rightEntropy) > bestGainInEntropy)
                {
                    const int j = sorter.GetUnSortedIndex(sortedIndex+1);
                    bestGainInEntropy = (entropyStart - leftEntropy - rightEntropy);
                    bestThreshold = 0.5f * (featureValuesForTest[i] + featureValuesForTest[j]);

                    for(int c=0; c<mMaxClass; c++)
                    {
                        bestLeftClassLabelCounts[c] = leftClassLabelCounts[c];
                        bestRightClassLabelCounts[c] = rightClassLabelCounts[c];
                        bestLeftWeight = leftWeight;
                        bestRightWeight = rightWeight;
                    }

                }
                // printf("sortedIndex=%d i=%d entropyStart=%0.2f leftEntropy=%0.2f rightEntropy=%0.2f  bestGainInEntropy=%0.2f bestThreshold=%0.2f\n",
                //             sortedIndex, i, entropyStart, leftEntropy, rightEntropy, bestGainInEntropy, bestThreshold);
            }

        }
        impurityOut.Set(testIndex, 0, bestGainInEntropy);
        thresholdOut.Set(testIndex, 0, bestThreshold);
        childCountsOut.Set(testIndex, 0, bestLeftWeight);
        childCountsOut.Set(testIndex, 1, bestRightWeight);

        for(int c=0; c<mMaxClass; c++)
        {
            leftYsOut.Set(testIndex, c, bestLeftClassLabelCounts[c] / bestLeftWeight);
            rightYsOut.Set(testIndex, c, bestRightClassLabelCounts[c] / bestRightWeight );
        }
    }
}