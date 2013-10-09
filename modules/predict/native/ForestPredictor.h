#pragma once

#include <BufferCollection.h>
#include <BufferCollectionStack.h>
#include <Forest.h>
#include <Constants.h>
#include <PipelineStepI.h>

template <class FeatureBinding, class BufferTypes>
typename BufferTypes::Index nextChild(  const FeatureBinding& feature,
                    const Tree& tree,
                    const typename BufferTypes::Index nodeId,
                    const typename BufferTypes::Index index )
{
    const typename BufferTypes::FeatureValue splitpoint = tree.GetFloatFeatureParams().Get(nodeId, SPLIT_POINT_INDEX);
    const typename BufferTypes::FeatureValue featureValue = feature.FeatureValue(nodeId, index);
    bool goLeft = (featureValue > splitpoint);
    const typename BufferTypes::Index childDirection = goLeft ? 0 : 1;
    const typename BufferTypes::Index childNodeId = tree.GetPath().Get(nodeId, childDirection);
    return childNodeId;
}

template <class FeatureBinding, class BufferTypes>
typename BufferTypes::Index walkTree( const FeatureBinding& feature,
                  const Tree& tree,
                  const typename BufferTypes::Index nodeId,
                  const typename BufferTypes::Index index )
{
    const typename BufferTypes::Index childNodeId = nextChild<FeatureBinding,BufferTypes>( feature, tree, nodeId, index);
    if(childNodeId == NULL_CHILD)
    {
       return nodeId;
    }
    return walkTree<FeatureBinding,BufferTypes>(feature, tree, childNodeId, index);
}


template <class Feature, class Combiner, class BufferTypes>
class TemplateForestPredictor
{
public:
    TemplateForestPredictor( const Forest& forest, const Feature& feature, const Combiner& combiner, const PipelineStepI* preSteps );
    ~TemplateForestPredictor();

    void PredictLeafs(const BufferCollection& data, MatrixBufferTemplate<int>& leafsOut) const;
    void PredictYs(const BufferCollection& data, MatrixBufferTemplate<float>& ysOut);
    void PredictOobYs(const BufferCollection& data, MatrixBufferTemplate<float>& ysOut);
    void PredictYs(const BufferCollection& data, const VectorBufferTemplate<double>& treeWeights, MatrixBufferTemplate<float>& ysOut);
    void PredictOobYs(const BufferCollection& data, const VectorBufferTemplate<double>& treeWeights, MatrixBufferTemplate<float>& ysOut);
    void PredictYs(const BufferCollection& data, const VectorBufferTemplate<double>& treeWeights, const MatrixBufferTemplate<int>& leafs, MatrixBufferTemplate<float>& ysOut);
    void PredictOobYs(const BufferCollection& data, const VectorBufferTemplate<double>& treeWeights, const MatrixBufferTemplate<int>& leafs, MatrixBufferTemplate<float>& ysOut);

    void PredictLeafYs(const BufferCollection& data, MatrixBufferTemplate<float>& oobWeights, Tensor3BufferTemplate<float>& ysOut) const;

    void SetForest(const Forest& forest);
    Forest GetForest() const;
    void AddTree(const Tree& tree);

private:
    void PredictYsInternal(const BufferCollection& data, 
                            MatrixBufferTemplate<float>& ysOut, 
                            const VectorBufferTemplate<double>& treeWeights,
                            bool useOobIndices,
                            const MatrixBufferTemplate<int>* leafs);

    Forest mForest;
    Feature mFeature;
    Combiner mCombiner;
    const PipelineStepI* mPreSteps;
};

template <class Feature, class Combiner, class BufferTypes>
TemplateForestPredictor<Feature, Combiner, BufferTypes>::TemplateForestPredictor( const Forest& forest, const Feature& feature, const Combiner& combiner, const PipelineStepI* preSteps )
: mForest(forest)
, mFeature(feature)
, mCombiner(combiner)
, mPreSteps(preSteps->Clone())
{}

template <class Feature, class Combiner, class BufferTypes>
TemplateForestPredictor<Feature, Combiner, BufferTypes>::~TemplateForestPredictor()
{
    delete mPreSteps;
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictLeafs( const BufferCollection& data,
                                                                                  MatrixBufferTemplate<int>& leafsOut) const
{
    boost::mt19937 gen;
    gen.seed(0);

    const int numberOfTreesInForest = mForest.mTrees.size();
    BufferCollectionStack stack;
    stack.Push(&data);

    BufferCollection* perTreeBufferCollection = new BufferCollection[numberOfTreesInForest];
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTreesInForest);
    for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
    {
        BufferCollection& bc = perTreeBufferCollection[treeId];
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFeature.mFloatParamsBufferId, mForest.mTrees[treeId].GetFloatFeatureParams());
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mFeature.mIntParamsBufferId, mForest.mTrees[treeId].GetIntFeatureParams());
        mPreSteps->ProcessStep(stack, bc, gen, bc, 0);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    leafsOut.Resize(numberOfIndices, numberOfTreesInForest);

    for(typename BufferTypes::Index i=0; i<numberOfIndices; i++)
    {
        for(typename BufferTypes::Index treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            typename BufferTypes::Index leafNodeId = walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                                                            featureBindings[treeId], mForest.mTrees[treeId], 0, i);
            leafsOut.Set(i, treeId, leafNodeId);
        }
    }

    delete[] perTreeBufferCollection;
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictYs( const BufferCollection& data,
                                                                              MatrixBufferTemplate<float>& ysOut)
{
    VectorBufferTemplate<double> treeWeights(mForest.mTrees.size());
    treeWeights.SetAll(1.0);
    PredictYsInternal(data, ysOut, treeWeights, false, NULL);
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictOobYs( const BufferCollection& data,
                                                                              MatrixBufferTemplate<float>& ysOut)
{
    VectorBufferTemplate<double> treeWeights(mForest.mTrees.size());
    treeWeights.SetAll(1.0);
    PredictYsInternal(data, ysOut, treeWeights, true, NULL);
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictYs( const BufferCollection& data,
                                                                         const VectorBufferTemplate<double>& treeWeights,
                                                                         MatrixBufferTemplate<float>& ysOut)
{
    PredictYsInternal(data, ysOut, treeWeights, false, NULL);
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictOobYs( const BufferCollection& data,
                                                                            const VectorBufferTemplate<double>& treeWeights,
                                                                            MatrixBufferTemplate<float>& ysOut)
{
    PredictYsInternal(data, ysOut, treeWeights, true, NULL);
}


template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictYs( const BufferCollection& data,
                                                                         const VectorBufferTemplate<double>& treeWeights,
                                                                         const MatrixBufferTemplate<int>& leafs,
                                                                         MatrixBufferTemplate<float>& ysOut)
{
    PredictYsInternal(data, ysOut, treeWeights, false, &leafs);
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictOobYs( const BufferCollection& data,
                                                                            const VectorBufferTemplate<double>& treeWeights,
                                                                            const MatrixBufferTemplate<int>& leafs,
                                                                            MatrixBufferTemplate<float>& ysOut)
{
    PredictYsInternal(data, ysOut, treeWeights, true, &leafs);
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictYsInternal( const BufferCollection& data,
                                                                              MatrixBufferTemplate<float>& ysOut,
                                                                              const VectorBufferTemplate<double>& treeWeights,
                                                                              bool useOobIndices,
                                                                              const MatrixBufferTemplate<int>* leafs)
{
    boost::mt19937 gen;
    gen.seed(0);

    const int numberOfTreesInForest = mForest.mTrees.size();
    BufferCollectionStack stack;
    stack.Push(&data);

    std::vector<BufferCollection> perTreeBufferCollection(numberOfTreesInForest);
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTreesInForest);

    std::vector< const VectorBufferTemplate<typename BufferTypes::Index>* > oobIndices(numberOfTreesInForest);
    std::vector<typename BufferTypes::Index> currentOobOffset(numberOfTreesInForest);

    for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
    {
        const Tree& tree = mForest.mTrees[treeId];
        BufferCollection& bc = perTreeBufferCollection[treeId];

        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFeature.mFloatParamsBufferId, tree.GetFloatFeatureParams());
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mFeature.mIntParamsBufferId, tree.GetIntFeatureParams());
        mPreSteps->ProcessStep(stack, bc, gen, bc, 0);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();

        if(useOobIndices)
        {
            oobIndices[treeId] = tree.GetExtraInfo().GetBufferPtr< VectorBufferTemplate<typename BufferTypes::Index> >(OOB_INDICES);
            ASSERT(oobIndices[treeId]->IsSorted()) //Assuming OOB_INDICES have already been sorted
            currentOobOffset[treeId] = 0;
        }
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    ysOut.Resize(numberOfIndices, mCombiner.GetResultDim());

    for(typename BufferTypes::Index i=0; i<numberOfIndices; i++)
    {
        mCombiner.Reset();
        for(typename BufferTypes::Index treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            // Only include an datapoint if it is OOB or if we're not using OOB samples 
            const bool isOobIndex = useOobIndices && (oobIndices[treeId]->Get(currentOobOffset[treeId]) == i );
            ASSERT(!useOobIndices || (oobIndices[treeId]->Get(currentOobOffset[treeId]) < numberOfIndices))
            if(isOobIndex || !useOobIndices)
            {
                const Tree& tree = mForest.mTrees[treeId];
                typename BufferTypes::Index leafNodeId = (leafs != NULL) ? leafs->Get(i, treeId) :
                                                walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                                                    featureBindings[treeId], tree, 0, i);
                mCombiner.Combine(leafNodeId, tree.GetCounts().Get(leafNodeId), tree.GetYs(), treeWeights.Get(treeId));
            }

            if(isOobIndex)
            {
                currentOobOffset[treeId] = std::min(oobIndices[treeId]->GetN()-1, currentOobOffset[treeId]+1);
            }

        }
        mCombiner.WriteResult(i, ysOut);
    }
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::PredictLeafYs(const BufferCollection& data, 
                                                                            MatrixBufferTemplate<float>& oobWeights, 
                                                                            Tensor3BufferTemplate<float>& ysOut) const
{
    boost::mt19937 gen;
    gen.seed(0);

    const int numberOfTreesInForest = mForest.mTrees.size();
    BufferCollectionStack stack;
    stack.Push(&data);

    std::vector<BufferCollection> perTreeBufferCollection(numberOfTreesInForest);
    std::vector<typename Feature::FeatureBinding> featureBindings(numberOfTreesInForest);

    std::vector< bool > hasOobIndices(numberOfTreesInForest);
    std::vector< const VectorBufferTemplate<typename BufferTypes::Index>* > oobIndices(numberOfTreesInForest);
    std::vector<typename BufferTypes::Index> currentOobOffset(numberOfTreesInForest);


    for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
    {
        const Tree& tree = mForest.mTrees[treeId];
        BufferCollection& bc = perTreeBufferCollection[treeId];

        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFeature.mFloatParamsBufferId, tree.GetFloatFeatureParams());
        bc.AddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mFeature.mIntParamsBufferId, tree.GetIntFeatureParams());
        mPreSteps->ProcessStep(stack, bc, gen, bc, 0);

        stack.Push(&bc);
        featureBindings[treeId] = mFeature.Bind(stack);
        stack.Pop();

        hasOobIndices[treeId] = tree.GetExtraInfo().HasBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(OOB_INDICES); 

        if(hasOobIndices[treeId])
        {
            oobIndices[treeId] = tree.GetExtraInfo().GetBufferPtr< VectorBufferTemplate<typename BufferTypes::Index> >(OOB_INDICES);
            ASSERT(oobIndices[treeId]->IsSorted()) //Assuming OOB_INDICES have already been sorted
            currentOobOffset[treeId] = 0;
        }
    }

    const int numberOfIndices = featureBindings[0].GetNumberOfDatapoints();
    const int yDim = mCombiner.GetResultDim();
    oobWeights.Resize(numberOfIndices, numberOfTreesInForest);
    ysOut.Resize(yDim, numberOfIndices, numberOfTreesInForest);

    for(typename BufferTypes::Index i=0; i<numberOfIndices; i++)
    {
        for(typename BufferTypes::Index treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            const Tree& tree = mForest.mTrees[treeId];
            typename BufferTypes::Index leafNodeId = walkTree<typename Feature::FeatureBinding, BufferTypes>(
                                                                featureBindings[treeId], tree, 0, i);

            for(int c=0; c<yDim; c++)
            {
                ysOut.Set(c, i, treeId, tree.GetYs().Get(leafNodeId, c));
            }

            float oobWeight = 1.0f;
            if(oobIndices[treeId])
            {
                if(oobIndices[treeId]->Get(currentOobOffset[treeId]) == i)
                {
                    oobWeight = 1.0f;
                    currentOobOffset[treeId] = std::min(oobIndices[treeId]->GetN()-1, currentOobOffset[treeId]+1);
                }
                else
                {
                    oobWeight = 0.0f;
                }
            }
            oobWeights.Set(i, treeId, oobWeight);
        }
    }
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::SetForest(const Forest& forest) 
{
    mForest = forest;
}

template <class Feature, class Combiner, class BufferTypes>
Forest TemplateForestPredictor<Feature, Combiner, BufferTypes>::GetForest() const
{
    return mForest;
}

template <class Feature, class Combiner, class BufferTypes>
void TemplateForestPredictor<Feature, Combiner, BufferTypes>::AddTree(const Tree& tree)
{
    return mForest.AddTree(tree);
}


