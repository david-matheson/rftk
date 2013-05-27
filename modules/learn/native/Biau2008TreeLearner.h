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
#include "ActiveLeaf.h"

template <class FloatType, class IntType>
class Biau2008TreeLearner: public TreeLearnerI
{
public:
    Biau2008TreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                            const PipelineStepI* treeSteps,
                            const PipelineStepI* nodeSteps,
                            const SplitSelectorI<FloatType, IntType>* splitSelector,
                            const IntType maxNumberOfLeaves,
                            const IntType maxNumberOfSplitRetries);
    Biau2008TreeLearner(const Biau2008TreeLearner<FloatType, IntType> & other);

    virtual ~Biau2008TreeLearner();

    virtual TreeLearnerI* Clone() const;
    virtual void Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed ) const;

private:
    bool ProcessActiveLeaf( boost::mt19937& gen,
                              Tree& tree,
                              BufferCollectionStack& stack,
                              std::list< ActiveLeaf<FloatType, IntType> >& activeLeaves,
                              const int leafIndex ) const;

    const TrySplitCriteriaI* mTrySplitCriteria;
    const PipelineStepI* mTreeSteps;
    const PipelineStepI* mNodeSteps;
    const SplitSelectorI<FloatType, IntType>* mSplitSelector;
    const IntType mMaxNumberOfSplits;
    const IntType mMaxNumberOfSplitRetries;
};

template <class FloatType, class IntType>
Biau2008TreeLearner<FloatType, IntType>::Biau2008TreeLearner( const TrySplitCriteriaI* trySplitCriteria,
                                                                const PipelineStepI* treeSteps,
                                                                const PipelineStepI* nodeSteps,
                                                                const SplitSelectorI<FloatType, IntType>* splitSelector,
                                                                const IntType maxNumberOfLeaves,
                                                                const IntType maxNumberOfSplitRetries)
: mTrySplitCriteria( trySplitCriteria->Clone() )
, mTreeSteps( treeSteps->Clone() )
, mNodeSteps( nodeSteps->Clone() )
, mSplitSelector( splitSelector->Clone() )
, mMaxNumberOfSplits( maxNumberOfLeaves )
, mMaxNumberOfSplitRetries( maxNumberOfSplitRetries )
{}

template <class FloatType, class IntType>
Biau2008TreeLearner<FloatType, IntType>::Biau2008TreeLearner(const Biau2008TreeLearner<FloatType, IntType> & other)
: mTrySplitCriteria( other.mTrySplitCriteria->Clone() )
, mTreeSteps( other.mTreeSteps->Clone() )
, mNodeSteps( other.mNodeSteps->Clone() )
, mSplitSelector( other.mSplitSelector->Clone() )
, mMaxNumberOfSplits( other.mMaxNumberOfSplits )
, mMaxNumberOfSplitRetries( other.mMaxNumberOfSplitRetries )
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
void Biau2008TreeLearner<FloatType, IntType>::Learn( BufferCollectionStack stack, Tree& tree, unsigned int seed ) const
{
    boost::mt19937 gen;
    gen.seed(seed);

    BufferCollection treeData;
    stack.Push(&treeData);
    mTreeSteps->ProcessStep(stack, treeData, gen);

    std::list< ActiveLeaf<FloatType, IntType> > activeLeaves;
    activeLeaves.push_back(ActiveLeaf<FloatType, IntType>(0, 0));

    // Select an active leaf uniformly at random
    int numberOfSplits = 0;
    int numberOfRetriesRemaining = mMaxNumberOfSplitRetries;
    while(numberOfSplits < mMaxNumberOfSplits 
          && activeLeaves.size() > 0
          && numberOfRetriesRemaining >= 0)
    {
        boost::uniform_int<> uniform_leaf(0, activeLeaves.size()-1);
        boost::variate_generator<boost::mt19937&,boost::uniform_int<> > var_uniform_leaf(gen, uniform_leaf);
        const int leafIndex =  var_uniform_leaf();
        const bool didSplit = ProcessActiveLeaf(gen, tree, stack, activeLeaves, leafIndex);
        if( didSplit )
        {
            numberOfSplits++; 
            numberOfRetriesRemaining = mMaxNumberOfSplitRetries;
        }
        else
        {
            numberOfRetriesRemaining--;
        }
    }
}

template <class FloatType, class IntType>
bool Biau2008TreeLearner<FloatType, IntType>::ProcessActiveLeaf( boost::mt19937& gen,
                                                              Tree& tree,
                                                              BufferCollectionStack& stack,
                                                              std::list< ActiveLeaf<FloatType, IntType> >& activeLeaves,
                                                              const int leafIndex ) const

{
    // Get the active leaf
    typename std::list< ActiveLeaf<FloatType, IntType> >::iterator iter = activeLeaves.begin();
    for(int i=0; i<leafIndex; i++)
    {
        ++iter;
    }
    ActiveLeaf<FloatType, IntType>& activeLeaf = *iter;

    stack.Push(&activeLeaf.mSplitBufferCollection);
    BufferCollection nodeData;
    stack.Push(&nodeData);
    mNodeSteps->ProcessStep(stack, nodeData, gen);
    SplitSelectorInfo<FloatType, IntType> selectorInfo = mSplitSelector->ProcessSplits(stack, activeLeaf.mDepth);

    const bool ValidSplit = selectorInfo.ValidSplit(); //Incase all datapoints have the same value

    if(ValidSplit) 
    {
        const IntType leftNodeIndex = tree.NextNodeIndex();
        const IntType rightNodeIndex = tree.NextNodeIndex();

        selectorInfo.WriteToTree( activeLeaf.mNodeIndex, leftNodeIndex, rightNodeIndex,
                                  tree.mCounts, tree.mDepths, tree.mFloatFeatureParams, tree.mIntFeatureParams, tree.mYs);

        tree.mPath.Set(activeLeaf.mNodeIndex, 0, leftNodeIndex);
        tree.mPath.Set(activeLeaf.mNodeIndex, 1, rightNodeIndex);

        ActiveLeaf<FloatType, IntType> leftActiveLeaf(leftNodeIndex, activeLeaf.mDepth+1);
        ActiveLeaf<FloatType, IntType> rightActiveLeaf(rightNodeIndex, activeLeaf.mDepth+1);
        FloatType leftSize = std::numeric_limits<FloatType>::min();
        FloatType rightSize = std::numeric_limits<FloatType>::min();
        selectorInfo.SplitBuffers(leftActiveLeaf.mSplitBufferCollection, rightActiveLeaf.mSplitBufferCollection, leftSize, rightSize);

        if(mTrySplitCriteria->TrySplit(leftActiveLeaf.mDepth, leftSize))
        {
            activeLeaves.push_back(leftActiveLeaf);
        }
        if(mTrySplitCriteria->TrySplit(rightActiveLeaf.mDepth, rightSize))
        {
            activeLeaves.push_back(rightActiveLeaf);
        }

        // Remove the node
        const int numberOfActiveLeaves = activeLeaves.size();
        UNUSED_PARAM(numberOfActiveLeaves);
        activeLeaves.erase(iter);
        ASSERT(numberOfActiveLeaves-1 == activeLeaves.size()); //check that it actually decreased in size
    }

    stack.Pop(); //stack.Push(activeLeaf.mSplitBufferCollection)
    stack.Pop(); //stack.Push(&nodeData);

    return ValidSplit;
}