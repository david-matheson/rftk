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
#include <SplitSelectorI.h>

#include "ActiveLeaf.h"
#include "ProbabilityOfErrorFrontierQueue.h"


// ----------------------------------------------------------------------------
//
// OnlineForestLearner learns a forest online by storing stats of active leafs
//
// ----------------------------------------------------------------------------
template <class Feature, class EstimatorUpdater, class ProbabilityOfError, class BufferTypes>
class OnlineForestLearner
{
public:

    OnlineForestLearner( const TrySplitCriteriaI* trySplitCriteria,
                        const PipelineStepI* treeSteps,
                        const PipelineStepI* initNodeSteps,
                        const PipelineStepI* statsUpdateNodeSteps,
                        const PipelineStepI* impurityUpdateNodeSteps,
                        const int impurityUpdatePeriod,
                        const SplitSelectorI<BufferTypes>* splitSelector,
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
    OnlineForestLearner(const OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError,BufferTypes>& rhs);
    OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError,BufferTypes> & operator= 
                      (const OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError,BufferTypes> & other);

    void UpdateActiveFrontier();

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mInitNodeSteps;
    const PipelineStepI* mStatsUpdateNodeSteps;
    const PipelineStepI* mImpurityUpdateNodeSteps;
    const int mImpurityUpdatePeriod;
    const SplitSelectorI<BufferTypes>* mSplitSelector;

    const int mMaxFrontierSize;

    Forest mForest;
    ProbabilityOfErrorFrontierQueue<ProbabilityOfError> mFrontierQueue;
    std::map< std::pair<int, int>, ActiveOnlineLeaf > mActiveFrontierLeaves;

    const BufferId mIndicesBufferId;
    const BufferId mWeightsBufferId;

    const Feature mPredictFeature;
    const EstimatorUpdater mEstimatorUpdater;
};

template <class Feature, class EstimatorUpdater, class ProbabilityOfError, class BufferTypes>
OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, BufferTypes>
::OnlineForestLearner( const TrySplitCriteriaI* trySplitCriteria,
                      const PipelineStepI* treeSteps,
                      const PipelineStepI* initNodeSteps,
                      const PipelineStepI* statsUpdateNodeSteps,
                      const PipelineStepI* impurityUpdateNodeSteps,
                      const int impurityUpdatePeriod,
                      const SplitSelectorI<BufferTypes>* splitSelector,
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

template <class Feature, class EstimatorUpdater, class ProbabilityOfError, class BufferTypes>
OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, BufferTypes>
::~OnlineForestLearner()
{
    typedef std::map< std::pair<int, int>, ActiveOnlineLeaf >::iterator it_type;
    for(it_type iterator = mActiveFrontierLeaves.begin(); iterator != mActiveFrontierLeaves.end(); ++iterator)
    {
        delete iterator->second.mNodeData;
        iterator->second.mNodeData = NULL;
    }
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError, class BufferTypes>
Forest OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, BufferTypes>
::Learn( const BufferCollection& data )
{
    boost::mt19937 gen;
    gen.seed( static_cast<unsigned int>(std::time(NULL)) );

    const int numberOfTrees = mForest.mTrees.size();

    std::vector<BufferCollectionStack> forestStacks(numberOfTrees);
    std::vector<BufferCollection> forestBcs(numberOfTrees);
    std::vector< VectorBufferTemplate<typename BufferTypes::Index>* > forestIndices(numberOfTrees);
    std::vector< VectorBufferTemplate<typename BufferTypes::SourceContinuous> const* > forestWeights(numberOfTrees);
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTrees);
    std::vector<typename EstimatorUpdater::BindedEstimatorUpdater> estimatorUpdaterBindings(numberOfTrees);
    for(int treeIndex=0; treeIndex<numberOfTrees; treeIndex++)
    {
        Tree& tree = mForest.mTrees[treeIndex];
        BufferCollectionStack& treeStack = forestStacks[treeIndex];
        treeStack.Push(&data);
        BufferCollection& treeBc = forestBcs[treeIndex];
        treeStack.Push(&treeBc);
        mTreeSteps->ProcessStep(treeStack, treeBc, gen, tree.GetExtraInfo(), 0);
        forestIndices[treeIndex] = treeBc.GetBufferPtr< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId);
        forestWeights[treeIndex] = treeStack.GetBufferPtr< VectorBufferTemplate<typename BufferTypes::SourceContinuous> >(mWeightsBufferId);
        treeBc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mPredictFeature.mFloatParamsBufferId, tree.GetFloatFeatureParams());
        treeBc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mPredictFeature.mIntParamsBufferId, tree.GetIntFeatureParams());
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

            VectorBufferTemplate<typename BufferTypes::Index>* indices = forestIndices[treeIndex];
            indices->Resize(1);
            indices->Set(0, sampleIndex);

            const VectorBufferTemplate<typename BufferTypes::SourceContinuous>* weights = forestWeights[treeIndex];
            const typename BufferTypes::SourceContinuous sampleWeight = weights->Get(sampleIndex);

            if(sampleWeight <= typename BufferTypes::SourceContinuous(0)) continue;

            mFrontierQueue.IncrDatapoints(treeIndex, static_cast<long long>(sampleWeight));

            const int nodeIndex = walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                             featureBindings[treeIndex], tree, 0, 0 );
            const int depth = tree.GetDepths().Get(nodeIndex);

            estimatorUpdaterBindings[treeIndex].UpdateEstimator(tree, nodeIndex, sampleIndex);
            tree.GetCounts().Incr(nodeIndex, sampleWeight);

            // Update active node stats
            std::pair<int,int> treeNodeKey = std::make_pair(treeIndex, nodeIndex);
            if( mActiveFrontierLeaves.find(treeNodeKey) != mActiveFrontierLeaves.end() )
            {
                ActiveOnlineLeaf& leaf = mActiveFrontierLeaves[treeNodeKey];
                stack.Push(leaf.mNodeData);
                if( !leaf.mIsInitialized )
                {
                    mInitNodeSteps->ProcessStep(stack, *leaf.mNodeData, gen, tree.GetExtraInfo(), nodeIndex);
                    leaf.mIsInitialized = true;
                }

                mStatsUpdateNodeSteps->ProcessStep(stack, *leaf.mNodeData, gen, tree.GetExtraInfo(), nodeIndex);
                leaf.mDatapointsSinceLastImpurityUpdate++;

                if( leaf.mDatapointsSinceLastImpurityUpdate >= mImpurityUpdatePeriod )
                {
                    mImpurityUpdateNodeSteps->ProcessStep(stack, *leaf.mNodeData, gen, tree.GetExtraInfo(), nodeIndex);
                    leaf.mDatapointsSinceLastImpurityUpdate = 0;

                    // Split the node
                    SplitSelectorInfo<BufferTypes> selectorInfo = mSplitSelector->ProcessSplits(stack, depth, tree.GetExtraInfo(), nodeIndex);
                    if( selectorInfo.ValidSplit() )
                    {
                        const int leftNodeIndex = tree.NextNodeIndex();
                        const int rightNodeIndex = tree.NextNodeIndex();

                        selectorInfo.WriteToTree( nodeIndex, leftNodeIndex, rightNodeIndex,
                                                  tree.GetCounts(), tree.GetDepths(), tree.GetFloatFeatureParams(), tree.GetIntFeatureParams(), tree.GetYs());

                        tree.GetPath().Set(nodeIndex, LEFT_CHILD, leftNodeIndex);
                        tree.GetPath().Set(nodeIndex, RIGHT_CHILD, rightNodeIndex);

                        // Remove split node
                        mActiveFrontierLeaves.erase(treeNodeKey);
                        delete leaf.mNodeData;

                        // Add children to the queue
                        if( mTrySplitCriteria->TrySplit(depth, std::numeric_limits<int>::max(), tree.GetExtraInfo(), nodeIndex, true) )
                        {
                            mFrontierQueue.ProcessSplit(mForest, treeIndex, nodeIndex, leftNodeIndex, rightNodeIndex);
                        }

                        // Update the frontier priority queue
                        UpdateActiveFrontier();

                        // Update the forest params used for prediction since it has changed
                        BufferCollection& bc = forestBcs[treeIndex];

                        MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>* featureFloatParams =
                                bc.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mPredictFeature.mFloatParamsBufferId);
                        *featureFloatParams = tree.GetFloatFeatureParams();

                        MatrixBufferTemplate<typename BufferTypes::ParamsInteger>* featureIntParams =
                                bc.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mPredictFeature.mIntParamsBufferId);
                        *featureIntParams = tree.GetIntFeatureParams();
                    }
                }
                stack.Pop(); //stack.Push(leaf.mNodeData);
            }
        }
    }
    return mForest;
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError, class BufferTypes>
Forest OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, BufferTypes>
::GetForest() const
{
    return mForest;
}

template <class Feature, class EstimatorUpdater, class ProbabilityOfError, class BufferTypes>
void OnlineForestLearner<Feature, EstimatorUpdater, ProbabilityOfError, BufferTypes>
::UpdateActiveFrontier()
{
    // Add a new active split to the frontier when there is space and there is a candidate in the queue
    while( mActiveFrontierLeaves.size() < static_cast<size_t>(mMaxFrontierSize) && !mFrontierQueue.IsEmpty() )
    {
        std::pair<int,int> treeNodeKey = mFrontierQueue.PopBest(mForest);
        mActiveFrontierLeaves[treeNodeKey] = ActiveOnlineLeaf(new BufferCollection());
    }
}