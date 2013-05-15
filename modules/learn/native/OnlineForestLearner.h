#pragma once

#include <set>
#include <utility>
#include <vector>
#include <limits>
#include <ctime>

#include <VectorBuffer.h>
#include <MatrixBuffer.h>
#include <Tensor3Buffer.h>
#include <BufferCollectionStack.h>
#include <Tree.h>
#include <Forest.h>
#include <ForestPredictor.h>
#include <TrySplitCriteriaI.h>
#include <PipelineStepI.h>
#include <SplitSelector.h>

#include "ActiveLeaf.h"
#include "ProbabilityOfErrorFrontierQueue.h"


// ----------------------------------------------------------------------------
//
// OnlineForestLearner learns a forest online by storing stats of active leafs
//
// ----------------------------------------------------------------------------
template <class Feature, class EstimatorUpdater, class ProbabilityOfError,  class FloatType, class IntType>
class OnlineForestLearner
{
public:

    OnlineForestLearner( const TrySplitCriteriaI* trySplitCriteria,
                        const PipelineStepI* treeSteps,
                        const PipelineStepI* initNodeSteps,
                        const PipelineStepI* statsUpdateNodeSteps,
                        const PipelineStepI* impurityUpdateNodeSteps,
                        const int impurityUpdatePeriod,
                        const SplitSelectorI<FloatType, IntType>* splitSelector,
                        int maxFrontierSize,
                        int numberOfTrees,
                        int maxIntParamsDim,
                        int maxFloatParamsDim,
                        int maxEstimatorDim,
                        const BufferId& indicesBufferId,
                        const BufferId& weightsBufferId,
                        const Feature& predictFeature,
                        const EstimatorUpdater& estimatorParamsUpdater );
    ~OnlineForestLearner();

    Forest Learn(const BufferCollection& data);
    Forest GetForest() const;

private:
    void UpdateActiveFrontier();

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mInitNodeSteps;
    const PipelineStepI* mStatsUpdateNodeSteps;
    const PipelineStepI* mImpurityUpdateNodeSteps;
    const int mImpurityUpdatePeriod;
    const SplitSelectorI<FloatType, IntType>* mSplitSelector;

    const int mMaxFrontierSize;

    Forest mForest;
    ProbabilityOfErrorFrontierQueue<ProbabilityOfError> mFrontierQueue;
    std::map< std::pair<int, int>, ActiveLeaf > mActiveFrontierLeaves;

    const BufferId mIndicesBufferId;
    const BufferId mWeightsBufferId;

    const Feature mPredictFeature;
    const EstimatorUpdater mEstimatorUpdater;
};

template <class Feature, class EstimatorUpdater, class ProbabilityOfError,  class FloatType, class IntType>
OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, FloatType, IntType>
::OnlineForestLearner( const TrySplitCriteriaI* trySplitCriteria,
                      const PipelineStepI* treeSteps,
                      const PipelineStepI* initNodeSteps,
                      const PipelineStepI* statsUpdateNodeSteps,
                      const PipelineStepI* impurityUpdateNodeSteps,
                      const int impurityUpdatePeriod,
                      const SplitSelectorI<FloatType, IntType>* splitSelector,
                      int maxFrontierSize,
                      int numberOfTrees,
                      int maxIntParamsDim,
                      int maxFloatParamsDim,
                      int maxEstimatorDim,
                      const BufferId& indicesBufferId,
                      const BufferId& weightsBufferId,
                      const Feature& predictFeature,
                      const EstimatorUpdater& estimatorParamsUpdater  )
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mInitNodeSteps( initNodeSteps->Clone() )
, mStatsUpdateNodeSteps( statsUpdateNodeSteps->Clone() )
, mImpurityUpdateNodeSteps( impurityUpdateNodeSteps->Clone() )
, mImpurityUpdatePeriod(impurityUpdatePeriod)
, mSplitSelector( splitSelector->Clone() )
, mMaxFrontierSize(maxFrontierSize)
, mForest( numberOfTrees, 1, maxIntParamsDim, maxFloatParamsDim, maxEstimatorDim )
, mFrontierQueue(numberOfTrees)
, mActiveFrontierLeaves()
, mIndicesBufferId(indicesBufferId)
, mWeightsBufferId(weightsBufferId)
, mPredictFeature(predictFeature)
, mEstimatorUpdater(estimatorParamsUpdater)
{
    UpdateActiveFrontier();
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError,  class FloatType, class IntType>
OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, FloatType, IntType>
::~OnlineForestLearner()
{
    typedef std::map< std::pair<int, int>, ActiveLeaf >::iterator it_type;
    for(it_type iterator = mActiveFrontierLeaves.begin(); iterator != mActiveFrontierLeaves.end(); ++iterator)
    {
        delete iterator->second.mNodeData;
        iterator->second.mNodeData = NULL;
    }
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError,  class FloatType, class IntType>
Forest OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, FloatType, IntType>
::Learn( const BufferCollection& data )
{
    boost::mt19937 gen;
    gen.seed( static_cast<unsigned int>(std::time(NULL)) );

    const int numberOfTrees = mForest.mTrees.size();

    std::vector<BufferCollectionStack> forestStacks(numberOfTrees);
    std::vector<BufferCollection> forestBcs(numberOfTrees);
    std::vector< VectorBufferTemplate<IntType>* > forestIndices(numberOfTrees);
    std::vector< VectorBufferTemplate<FloatType> const* > forestWeights(numberOfTrees);
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTrees);
    std::vector<typename EstimatorUpdater::BindedEstimatorUpdater> estimatorUpdaterBindings(numberOfTrees);
    for(int treeIndex=0; treeIndex<numberOfTrees; treeIndex++)
    {
        Tree& tree = mForest.mTrees[treeIndex];
        BufferCollectionStack& treeStack = forestStacks[treeIndex];
        treeStack.Push(&data);
        BufferCollection& treeBc = forestBcs[treeIndex];
        treeStack.Push(&treeBc);
        mTreeSteps->ProcessStep(treeStack, treeBc, gen);
        forestIndices[treeIndex] = treeBc.GetBufferPtr< VectorBufferTemplate<IntType> >(mIndicesBufferId);
        forestWeights[treeIndex] = treeStack.GetBufferPtr< VectorBufferTemplate<FloatType> >(mWeightsBufferId);
        treeBc.AddBuffer< MatrixBufferTemplate<FloatType> >(mPredictFeature.mFloatParamsBufferId, tree.mFloatFeatureParams);
        treeBc.AddBuffer< MatrixBufferTemplate<IntType> >(mPredictFeature.mIntParamsBufferId, tree.mIntFeatureParams);
        featureBindings[treeIndex] = mPredictFeature.Bind(treeStack);
        estimatorUpdaterBindings[treeIndex] = mEstimatorUpdater.Bind(treeStack);
    }

    const int numberOfSamples = forestWeights[0]->GetN();

    for(int sampleIndex=0; sampleIndex<numberOfSamples; sampleIndex++)
    {
        for(int treeIndex=0; treeIndex<numberOfTrees; treeIndex++)
        {
            Tree& tree = mForest.mTrees[treeIndex];
            BufferCollectionStack& stack = forestStacks[treeIndex];

            VectorBufferTemplate<IntType>* indices = forestIndices[treeIndex];
            indices->Resize(1);
            indices->Set(0, sampleIndex);

            const VectorBufferTemplate<FloatType>* weights = forestWeights[treeIndex];
            const FloatType sampleWeight = weights->Get(sampleIndex);

            if(sampleWeight <= FloatType(0)) continue;

            mFrontierQueue.IncrDatapoints(treeIndex, static_cast<long long>(sampleWeight));

            const int nodeIndex = walkTree<typename Feature::FeatureBinding, FloatType, IntType>(
                                             featureBindings[treeIndex], tree, 0, 0 );
            const int depth = tree.mDepths.Get(nodeIndex);

            estimatorUpdaterBindings[treeIndex].UpdateEstimator(tree, nodeIndex, sampleIndex);
            tree.mCounts.Incr(nodeIndex, sampleWeight);

            // Update active node stats
            std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, nodeIndex);
            if( mActiveFrontierLeaves.find(treeNodeKey) != mActiveFrontierLeaves.end() )
            {
                ActiveLeaf leaf = mActiveFrontierLeaves[treeNodeKey];
                stack.Push(leaf.mNodeData);
                if( !leaf.mIsInitialized )
                {
                    mInitNodeSteps->ProcessStep(stack, *leaf.mNodeData, gen);
                    leaf.mIsInitialized = true;
                }

                mStatsUpdateNodeSteps->ProcessStep(stack, *leaf.mNodeData, gen);
                leaf.mDatapointsSinceLastImpurityUpdate++;

                if( leaf.mDatapointsSinceLastImpurityUpdate >= mImpurityUpdatePeriod )
                {
                    mImpurityUpdateNodeSteps->ProcessStep(stack, *leaf.mNodeData, gen);
                    leaf.mDatapointsSinceLastImpurityUpdate = 0;

                    // Split the node
                    SplitSelectorInfo<FloatType, IntType> selectorInfo = mSplitSelector->ProcessSplits(stack, depth);
                    if( selectorInfo.ValidSplit() )
                    {
                        const int leftNodeIndex = tree.NextNodeIndex();
                        const int rightNodeIndex = tree.NextNodeIndex();

                        selectorInfo.WriteToTree( nodeIndex, leftNodeIndex, rightNodeIndex,
                                                  tree.mCounts, tree.mDepths, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mYs);

                        tree.mPath.Set(nodeIndex, LEFT_CHILD, leftNodeIndex);
                        tree.mPath.Set(nodeIndex, RIGHT_CHILD, rightNodeIndex);

                        // Remove split node
                        mActiveFrontierLeaves.erase(treeNodeKey);
                        delete leaf.mNodeData;

                        // Add children to the queue
                        if( mTrySplitCriteria->TrySplit(depth, std::numeric_limits<int>::max()) )
                        {
                            mFrontierQueue.ProcessSplit(mForest, treeIndex, nodeIndex, leftNodeIndex, rightNodeIndex);
                        }

                        // Update the frontier priority queue
                        UpdateActiveFrontier();

                        // Update the forest params used for prediction since it has changed
                        BufferCollection& bc = forestBcs[treeIndex];

                        MatrixBufferTemplate<FloatType>* featureFloatParams =
                                bc.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mPredictFeature.mFloatParamsBufferId);
                        *featureFloatParams = tree.mFloatFeatureParams;

                        MatrixBufferTemplate<IntType>* featureIntParams =
                                bc.GetBufferPtr< MatrixBufferTemplate<IntType> >(mPredictFeature.mIntParamsBufferId);
                        *featureIntParams = tree.mIntFeatureParams;
                    }
                }
                stack.Pop(); //stack.Push(leaf.mNodeData);
            }
        }
    }
    return mForest;
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError,  class FloatType, class IntType>
Forest OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, FloatType, IntType>
::GetForest() const
{
    return mForest;
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError,  class FloatType, class IntType>
void OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, FloatType, IntType>
::UpdateActiveFrontier()
{
    // Add a new active split to the frontier when there is space and there is a candidate in the queue
    while( mActiveFrontierLeaves.size() < static_cast<size_t>(mMaxFrontierSize) && !mFrontierQueue.IsEmpty() )
    {
        std::pair<int,int> treeNodeKey = mFrontierQueue.PopBest(mForest);
        mActiveFrontierLeaves[treeNodeKey] = ActiveLeaf(new BufferCollection());
    }
}