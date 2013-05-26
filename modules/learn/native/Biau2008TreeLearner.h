#pragma once

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <limits>
#include <list>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollectionStack.h"
#include "Tree.h"
#include "TrySplitCriteriaI.h"
#include "PipelineStepI.h"
#include "SplitSelectorI.h"
#include "TreeLearnerI.h"

template <class FloatType, class IntType>
class Biau2008ActiveLeaf
{
public:
    Biau2008ActiveLeaf(const int nodeIndex,
                        const IntType depth)
    : mNodeIndex(nodeIndex)
    , mDepth(depth)
    , mIndices()
    {}

    const int mNodeIndex;
    const IntType mDepth;
    BufferCollection mIndices;
};

template <class FloatType, class IntType>
class Biau2008TreeLearner: public TreeLearnerI
{
public:
    Biau2008TreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                            const PipelineStepI* treeSteps,
                            const PipelineStepI* nodeSteps,
                            const SplitSelectorI<FloatType, IntType>* splitSelector,
                            const IntType maxNumberOfLeaves);
    Biau2008TreeLearner(const Biau2008TreeLearner<FloatType, IntType> & other);

    virtual ~Biau2008TreeLearner();

    virtual TreeLearnerI* Clone() const;
    virtual void Learn( const BufferCollection& data, Tree& tree, unsigned int seed ) const;

private:
    void ProcessActiveLeaf( boost::mt19937& gen,
                              Tree& tree,
                              BufferCollectionStack& stack,
                              std::list< Biau2008ActiveLeaf<FloatType, IntType> >& activeLeaves,
                              const int leafIndex ) const;

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mNodeSteps;
    const SplitSelectorI<FloatType, IntType>* mSplitSelector;
    const IntType mMaxNumberOfLeaves;
};

template <class FloatType, class IntType>
Biau2008TreeLearner<FloatType, IntType>::Biau2008TreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                                                                const PipelineStepI* treeSteps,
                                                                const PipelineStepI* nodeSteps,
                                                                const SplitSelectorI<FloatType, IntType>* splitSelector,
                                                                const IntType maxNumberOfLeaves)
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mNodeSteps( nodeSteps->Clone() )
, mSplitSelector( splitSelector->Clone() )
, mMaxNumberOfLeaves( maxNumberOfLeaves )
{}

template <class FloatType, class IntType>
Biau2008TreeLearner<FloatType, IntType>::Biau2008TreeLearner(const Biau2008TreeLearner<FloatType, IntType> & other)
: mTrySplitCriteria( other.mTrySplitCriteria->Clone() )
, mTreeSteps( other.mTreeSteps->Clone() )
, mNodeSteps( other.mNodeSteps->Clone() )
, mSplitSelector( other.mSplitSelector->Clone() )
, mMaxNumberOfLeaves( other.mMaxNumberOfLeaves )
{
}

template <class FloatType, class IntType>
Biau2008TreeLearner<FloatType, IntType>::~Biau2008TreeLearner()
{
    delete mTrySplitCriteria;
    delete mTreeSteps;
    delete mNodeSteps;
    delete mSplitSelector;
}

template <class FloatType, class IntType>
TreeLearnerI* Biau2008TreeLearner<FloatType, IntType>::Clone() const
{
    TreeLearnerI* clone = new Biau2008TreeLearner(*this);
    return clone;
}


template <class FloatType, class IntType>
void Biau2008TreeLearner<FloatType, IntType>::Learn( const BufferCollection& data, Tree& tree, unsigned int seed ) const
{
    boost::mt19937 gen;
    gen.seed(seed);

    BufferCollectionStack stack;
    stack.Push(&data);
    BufferCollection treeData;
    stack.Push(&treeData);
    mTreeSteps->ProcessStep(stack, treeData, gen);

    std::list< Biau2008ActiveLeaf<FloatType, IntType> > activeLeaves;
    activeLeaves.push_back(Biau2008ActiveLeaf<FloatType, IntType>(0, 0));

    // Select an active leaf uniformly at random
    int numberOfLeaves = 1;
    while(numberOfLeaves <= mMaxNumberOfLeaves && activeLeaves.size() > 0)
    {
        boost::uniform_int<> uniform_leaf(0, activeLeaves.size()-1);
        boost::variate_generator<boost::mt19937&,boost::uniform_int<> > var_uniform_leaf(gen, uniform_leaf);
        const int leafIndex =  var_uniform_leaf();
        ProcessActiveLeaf(gen, tree, stack, activeLeaves, leafIndex);
        numberOfLeaves++; //remove the parent and add two new leaves (-1+2=1)
    }
}

template <class FloatType, class IntType>
void Biau2008TreeLearner<FloatType, IntType>::ProcessActiveLeaf( boost::mt19937& gen,
                                                              Tree& tree,
                                                              BufferCollectionStack& stack,
                                                              std::list< Biau2008ActiveLeaf<FloatType, IntType> >& activeLeaves,
                                                              const int leafIndex ) const

{
    // Get the active leaf
    typename std::list< Biau2008ActiveLeaf<FloatType, IntType> >::iterator iter = activeLeaves.begin();
    for(int i=0; i<leafIndex; i++)
    {
        ++iter;
    }
    Biau2008ActiveLeaf<FloatType, IntType>& activeLeaf = *iter;

    stack.Push(&activeLeaf.mIndices);
    BufferCollection nodeData;
    stack.Push(&nodeData);
    mNodeSteps->ProcessStep(stack, nodeData, gen);
    SplitSelectorInfo<FloatType, IntType> selectorInfo = mSplitSelector->ProcessSplits(stack, activeLeaf.mDepth);

    if(selectorInfo.ValidSplit()) //Incase all datapoints have the same value
    {
        const IntType leftNodeIndex = tree.NextNodeIndex();
        const IntType rightNodeIndex = tree.NextNodeIndex();

        selectorInfo.WriteToTree( activeLeaf.mNodeIndex, leftNodeIndex, rightNodeIndex,
                                  tree.mCounts, tree.mDepths, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mYs);

        tree.mPath.Set(activeLeaf.mNodeIndex, 0, leftNodeIndex);
        tree.mPath.Set(activeLeaf.mNodeIndex, 1, rightNodeIndex);

        Biau2008ActiveLeaf<FloatType, IntType> leftActiveLeaf(leftNodeIndex, activeLeaf.mDepth+1);
        Biau2008ActiveLeaf<FloatType, IntType> rightActiveLeaf(rightNodeIndex, activeLeaf.mDepth+1);
        FloatType leftSize = std::numeric_limits<FloatType>::min();
        FloatType rightSize = std::numeric_limits<FloatType>::min();
        selectorInfo.SplitBuffers(leftActiveLeaf.mIndices, rightActiveLeaf.mIndices, leftSize, rightSize);

        if(mTrySplitCriteria->TrySplit(leftActiveLeaf.mDepth, leftSize))
        {
            activeLeaves.push_back(leftActiveLeaf);
        }
        if(mTrySplitCriteria->TrySplit(rightActiveLeaf.mDepth, rightSize))
        {
            activeLeaves.push_back(rightActiveLeaf);
        }
    }

    const int numberOfActiveLeaves = activeLeaves.size();
    UNUSED_PARAM(numberOfActiveLeaves);
    activeLeaves.erase(iter);
    ASSERT(numberOfActiveLeaves-1 == activeLeaves.size()); //check that it actually decreased in size

    stack.Pop(); //stack.Push(activeLeaf.mIndices)
    stack.Pop(); //stack.Push(&nodeData);
}