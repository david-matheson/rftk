#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <limits>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "Tree.h"
#include "TrySplitCriteriaI.h"
#include "PipelineStepI.h"
#include "SplitSelectorI.h"
#include "TreeLearnerI.h"


template <class BufferTypes>
class DepthFirstTreeLearner: public TreeLearnerI
{
public:
    DepthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                            const PipelineStepI* treeSteps,
                            const PipelineStepI* nodeSteps,
                            const SplitSelectorI<BufferTypes>* splitSelector);
    DepthFirstTreeLearner(const DepthFirstTreeLearner<BufferTypes> & other);

    virtual ~DepthFirstTreeLearner();

    virtual TreeLearnerI* Clone() const;
    virtual void Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed ) const;

private:
    void ProcessNode( boost::mt19937& gen,
                      Tree& tree,
                      BufferCollectionStack& stack,
                      typename BufferTypes::Index nodeIndex,
                      typename BufferTypes::Index depth,
                      typename BufferTypes::DatapointCounts nodeSize ) const;

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mNodeSteps;
    const SplitSelectorI<BufferTypes>* mSplitSelector;
};

template <class BufferTypes>
DepthFirstTreeLearner<BufferTypes>::DepthFirstTreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                                                                  const PipelineStepI* treeSteps,
                                                                  const PipelineStepI* nodeSteps,
                                                                  const SplitSelectorI<BufferTypes>* splitSelector)
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mNodeSteps( nodeSteps->Clone() )
, mSplitSelector( splitSelector->Clone() )
{}

template <class BufferTypes>
DepthFirstTreeLearner<BufferTypes>::DepthFirstTreeLearner(const DepthFirstTreeLearner<BufferTypes> & other)
: mTrySplitCriteria( other.mTrySplitCriteria->Clone() )
, mTreeSteps( other.mTreeSteps->Clone() )
, mNodeSteps( other.mNodeSteps->Clone() )
, mSplitSelector( other.mSplitSelector->Clone() )
{
}

template <class BufferTypes>
DepthFirstTreeLearner<BufferTypes>::~DepthFirstTreeLearner()
{
    delete mTrySplitCriteria;
    delete mTreeSteps;
    delete mNodeSteps;
    delete mSplitSelector;
}

template <class BufferTypes>
TreeLearnerI* DepthFirstTreeLearner<BufferTypes>::Clone() const
{
    TreeLearnerI* clone = new DepthFirstTreeLearner(*this);
    return clone;
}


template <class BufferTypes>
void DepthFirstTreeLearner<BufferTypes>::Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed ) const
{
    boost::mt19937 gen;
    gen.seed(seed);

    BufferCollection treeData;
    stack.Push(&treeData);
    mTreeSteps->ProcessStep(stack, treeData, gen);

    //emptyIndicesCollection is pushed so subsequent leftIndicesBufCol and rightIndicesBufCol
    //can be popped before adding the next layer down
    BufferCollection emptyIndicesCollection;
    stack.Push(&emptyIndicesCollection);
    ProcessNode(gen, tree, stack, 0, 0, std::numeric_limits<typename BufferTypes::DatapointCounts>::max());
}

template <class BufferTypes>
void DepthFirstTreeLearner<BufferTypes>::ProcessNode( boost::mt19937& gen,
                                                              Tree& tree, BufferCollectionStack& stack,
                                                              typename BufferTypes::Index nodeIndex,
                                                              typename BufferTypes::Index depth,
                                                              typename BufferTypes::DatapointCounts nodeSize ) const
{
    if(mTrySplitCriteria->TrySplit(depth, nodeSize))
    {
        bool doSplit = false;
        BufferCollection leftIndicesBufCol;
        BufferCollection rightIndicesBufCol;
        typename BufferTypes::DatapointCounts leftSize = std::numeric_limits<typename BufferTypes::DatapointCounts>::min();
        typename BufferTypes::DatapointCounts rightSize = std::numeric_limits<typename BufferTypes::DatapointCounts>::min();
        typename BufferTypes::Index leftNodeIndex = -1;
        typename BufferTypes::Index rightNodeIndex = -1;

        // Using a nested block so memory is freed before recursing
        {
            BufferCollection nodeData;
            stack.Push(&nodeData);
            mNodeSteps->ProcessStep(stack, nodeData, gen);
            SplitSelectorInfo<BufferTypes> selectorInfo = mSplitSelector->ProcessSplits(stack, depth);
            doSplit = selectorInfo.ValidSplit();
            if(doSplit)
            {
                leftNodeIndex = tree.NextNodeIndex();
                rightNodeIndex = tree.NextNodeIndex();

                selectorInfo.WriteToTree( nodeIndex, leftNodeIndex, rightNodeIndex,
                                          tree.mCounts, tree.mDepths, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mYs);

                tree.mPath.Set(nodeIndex, 0, leftNodeIndex);
                tree.mPath.Set(nodeIndex, 1, rightNodeIndex);

                selectorInfo.SplitBuffers(leftIndicesBufCol, rightIndicesBufCol, leftSize, rightSize);
            }
            stack.Pop(); //stack.Push(nodeData);
        }

        if( doSplit )
        {
            stack.Pop(); //stack.Push(&emptyIndicesCollection); or stack.Push(&leftIndicesBufCol); or stack.Push(&rightIndicesBufCol);

            stack.Push(&leftIndicesBufCol);
            ProcessNode(gen, tree, stack, leftNodeIndex, depth+1, leftSize);

            stack.Pop(); //stack.Push(&emptyIndicesCollection); or stack.Push(&leftIndicesBufCol); or stack.Push(&rightIndicesBufCol);

            stack.Push(&rightIndicesBufCol);
            ProcessNode(gen, tree, stack, rightNodeIndex, depth+1, rightSize);
        }
    }
}