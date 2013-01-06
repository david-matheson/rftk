#include "assert_util.h"
#include "MatrixBuffer.h"
#include "FeatureTypes.h"
#include "Forest.h"
#include "VecPredict.h"


VecForestPredictor::VecForestPredictor( const Forest& forest )
: mForest(forest)
{}

void VecForestPredictor::PredictLeafs(const MatrixBufferFloat& x, MatrixBufferInt& leafsOut)
{
    return VecPredictLeafs(mForest, x, leafsOut);
}

void VecForestPredictor::PredictYs(const MatrixBufferFloat& x, MatrixBufferFloat& ysOut)
{
    return VecPredictYs(mForest, x, ysOut);
}

int walkTree( const Tree& tree, int nodeId, const float* x );
int nextChild( const Tree& tree, int nodeId, const float* x );


void VecPredictLeafs(const Forest& forest, const MatrixBufferFloat& x, MatrixBufferInt& leafsOut)
{
    const int numberOfSamples = x.GetM();
    const int numberOfTreesInForest = forest.mTrees.size();
    // Create new results buffer if it's not the right dimensions
    if( leafsOut.GetM() != numberOfSamples || leafsOut.GetN() != numberOfTreesInForest )
    {
        leafsOut = MatrixBufferInt(numberOfSamples, numberOfTreesInForest);
    }

    for(int i=0; i<numberOfSamples; i++)
    {
        const float* x_ptr = x.GetRowPtrUnsafe(i);
        for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            int leafNodeId = walkTree(forest.mTrees[treeId], 0, x_ptr);
            leafsOut.Set(i, treeId, leafNodeId);
        }
    }
}

void VecPredictYs(const Forest& forest, const MatrixBufferFloat& x, MatrixBufferFloat& ysOut)
{
    // Create new results buffer if it's not the right dimensions
    const int numberOfSamples = x.GetM();
    const int numberOfTreesInForest = forest.mTrees.size();
    const int yDim = forest.mTrees[0].mYs.GetN();
    if( ysOut.GetM() != numberOfSamples || ysOut.GetN() != yDim )
    {
        ysOut = MatrixBufferFloat(numberOfSamples, yDim);
    }
    // Reset predictions if the buffer is being reused
    ysOut.Zero();


    // Create a temp buffer for leaf node id (this requires all leaf node ids to be stored in memory)
    // If the number of samples (ie number of rows in x) is to large this might be an issue
    MatrixBufferInt leafNodeIds = MatrixBufferInt(numberOfSamples, forest.mTrees.size());
    VecPredictLeafs(forest, x, leafNodeIds);

    float invNumberTrees = 1.0 / static_cast<float>(numberOfTreesInForest);

    for(int i=0; i<numberOfSamples; i++)
    {
        for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            int leafNodeId = leafNodeIds.Get(i, treeId);
            for(int c=0; c<yDim; c++)
            {
                const float delta = forest.mTrees[treeId].mYs.Get(leafNodeId, c) * invNumberTrees;
                const float updatedValue = ysOut.Get(i, c) + delta;
                ysOut.Set(i, c, updatedValue);
            }
        }
    }
}

int walkTree( const Tree& tree, int nodeId, const float* x )
{
    const int childNodeId = nextChild( tree, nodeId, x);
    if(childNodeId == -1)
    {
       return nodeId;
    }
    return walkTree(tree, childNodeId, x);
}

int nextChild( const Tree& tree, int nodeId, const float* x )
{
    // First int param is which feature to use
    const int featureType = tree.mIntFeatureParams.Get(nodeId, 0);
    // First float param is the threshold
    const float threshold = tree.mFloatFeatureParams.Get(nodeId, 0);
    bool testResult = false;
    switch( featureType )
    {
        case VEC_FEATURE_AXIS_ALIGNED:
        {
            int component = tree.mIntFeatureParams.Get(nodeId, 1);
            testResult = x[component] > threshold;
            break;
        }
        case VEC_FEATURE_PROJECTION:
        {
            float projectionValue = 0.0f;
            const int numberOfComponentsInProjection = tree.mIntFeatureParams.Get(nodeId, 1);
            for(int p=2; p<numberOfComponentsInProjection+2; p++)
            {
                const int componentId = tree.mIntFeatureParams.Get(nodeId, p);
                const float componentProjection = tree.mFloatFeatureParams.Get(nodeId, p);
                projectionValue += x[componentId] * componentProjection;
            }
            testResult = projectionValue > threshold;
            break;

        }
    }
    const int childDirection = testResult ? 0 : 1;
    const int childNodeId = tree.mPath.Get(nodeId, childDirection);
    return childNodeId;
}


