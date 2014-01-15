#include "BufferTypes.h"
#include "CreateDepthFirstLearner.h"
#include "SplitBuffersIndices.h"

DepthFirstTreeLearner<CdflBufferTypes_t> CreateDepthFirstLearner( BufferCollectionKey_t xs_key, 
                                                          BufferCollectionKey_t classes_key, 
                                                          int numberOfClasses, 
                                                          FeatureValueOrdering featureOrdering, 
                                                          double minNodeSize)
{
    // Don't try split if size is above a minimum
    MinNodeSizeCriteria trySplitCriteria(minNodeSize);

    // Tree steps
    std::vector<PipelineStepI*> treeSteps;
    AllSamplesStep<CdflBufferTypes_t, MatrixBufferTemplate<CdflBufferTypes_t::SourceContinuous > > allSamplesStep(xs_key);
    treeSteps.push_back(&allSamplesStep);
    VectorBufferTemplate<int> numberOfFeaturesBuffer = CreateVector1<int>(2);
    SetBufferStep< VectorBufferTemplate<int> > numberOfFeatures( numberOfFeaturesBuffer, WHEN_NEW );
    treeSteps.push_back(&numberOfFeatures);
    Pipeline treeStepsPipeline(treeSteps);

    // Node steps
    std::vector<PipelineStepI*> nodeSteps;
    AxisAlignedParamsStep<CdflBufferTypes_t, MatrixBufferTemplate<CdflBufferTypes_t::SourceContinuous > > featureParams(numberOfFeatures.OutputBufferId, xs_key);
    nodeSteps.push_back(&featureParams);
    LinearMatrixFeature<CdflBufferTypes_t, MatrixBufferTemplate<CdflBufferTypes_t::SourceContinuous > > feature(featureParams.FloatParamsBufferId,
                                                                          featureParams.IntParamsBufferId,
                                                                          allSamplesStep.IndicesBufferId,
                                                                          xs_key); 
    FeatureExtractorStep< LinearMatrixFeature<CdflBufferTypes_t, MatrixBufferTemplate<CdflBufferTypes_t::SourceContinuous > > > featureExtractor(feature, 
                                                                                                                                featureOrdering);
    nodeSteps.push_back(&featureExtractor);
    SliceBufferStep< CdflBufferTypes_t, VectorBufferTemplate<int> > sliceClasses(classes_key, allSamplesStep.IndicesBufferId);
    nodeSteps.push_back(&sliceClasses);
    SliceBufferStep< CdflBufferTypes_t, VectorBufferTemplate<float> > sliceWeights(allSamplesStep.WeightsBufferId, allSamplesStep.IndicesBufferId);  
    nodeSteps.push_back(&sliceWeights); 
    ClassInfoGainWalker<CdflBufferTypes_t> classInfoGainWalker(sliceWeights.SlicedBufferId, sliceClasses.SlicedBufferId, numberOfClasses);
    BestSplitpointsWalkingSortedStep< ClassInfoGainWalker<CdflBufferTypes_t> > bestSplitpointStep(classInfoGainWalker, 
                                                                                            featureExtractor.FeatureValuesBufferId,
                                                                                            featureOrdering,
                                                                                            AT_MIDPOINT);
    nodeSteps.push_back(&bestSplitpointStep); 
    Pipeline nodeStepsPipeline(nodeSteps);

    //Split Selector
    std::vector<SplitSelectorBuffers> splitBuffers;
    splitBuffers.push_back(SplitSelectorBuffers(bestSplitpointStep.ImpurityBufferId,
                                                bestSplitpointStep.SplitpointBufferId,
                                                bestSplitpointStep.SplitpointCountsBufferId,
                                                bestSplitpointStep.ChildCountsBufferId,
                                                bestSplitpointStep.LeftYsBufferId,
                                                bestSplitpointStep.RightYsBufferId,
                                                featureParams.FloatParamsBufferId,
                                                featureParams.IntParamsBufferId,
                                                featureExtractor.FeatureValuesBufferId,
                                                featureOrdering,
                                                &featureExtractor));
    const MinImpurityCriteria minImpurityCriteria(0.0);
    ClassEstimatorFinalizer<CdflBufferTypes_t> classFinalizer;
    SplitBuffersIndices<CdflBufferTypes_t> splitIndices(allSamplesStep.IndicesBufferId);
    SplitSelector<CdflBufferTypes_t> splitSelector(splitBuffers, &minImpurityCriteria, &classFinalizer, &splitIndices);
    
    return DepthFirstTreeLearner<CdflBufferTypes_t>(&trySplitCriteria, &treeStepsPipeline, &nodeStepsPipeline, &splitSelector);
}
