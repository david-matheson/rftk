#include <cstdio>

#include <limits>

#include "asserts/asserts.h"
#include "buffers/MatrixBuffer.h"
#include "features/FeatureTypes.h"
#include "features/ImgFeatures.h"
#include "forest_data/Forest.h"

#include "ForestPredictor.h"


ForestPredictor::ForestPredictor( const Forest& forest )
: mForest(forest)
{}

void ForestPredictor::PredictLeafs(BufferCollection& data,  const int numberOfindices, Int32MatrixBuffer& leafsOut)
{
    ForestPredictLeafs(mForest, data, numberOfindices, leafsOut);
}

void ForestPredictor::PredictYs(BufferCollection& data,  const int numberOfindices, Float32MatrixBuffer& ysOut)
{
    ForestPredictYs(mForest, data, numberOfindices, ysOut);
}

// void ForestPredictor::PredictMaxYs(BufferCollection& data, const int numberOfindices, Int32VectorBuffer& maxYsOut)
// {
//     ForestPredictMaxYs(mForest, data, numberOfindices, maxYsOut);
// }

void ForestPredictLeafs(const Forest& forest, BufferCollection& data, const int numberOfindices, Int32MatrixBuffer& leafsOut)
{
    const int numberOfTreesInForest = forest.mTrees.size();
    // Create new results buffer if it's not the right dimensions
    leafsOut.Resize(numberOfindices, numberOfTreesInForest);

    for(int i=0; i<numberOfindices; i++)
    {
        for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            int treeDepthOut = 0;
            int leafNodeId = walkTree(forest.mTrees[treeId], 0, data, i, treeDepthOut);
            leafsOut.Set(i, treeId, leafNodeId);
        }
    }
}

void ForestPredictYs(const Forest& forest, BufferCollection& data, const int numberOfindices, Float32MatrixBuffer& ysOut)
{
    // Create new results buffer if it's not the right dimensions
    const int numberOfTreesInForest = forest.mTrees.size();
    const int yDim = forest.mTrees[0].mYs.GetN();
    ysOut.Resize(numberOfindices, yDim);
    // Reset predictions if the buffer is being reused
    ysOut.Zero();

    const float invNumberTrees = 1.0 / static_cast<float>(numberOfTreesInForest);
    for(int i=0; i<numberOfindices; i++)
    {
        for(int treeId=0; treeId<numberOfTreesInForest; treeId++)
        {
            int treeDepthOut = 0;
            const int leafNodeId = walkTree(forest.mTrees[treeId], 0, data, i, treeDepthOut);
            for(int c=0; c<yDim; c++)
            {
                const float delta = forest.mTrees[treeId].mYs.Get(leafNodeId, c) * invNumberTrees;
                const float updatedValue = ysOut.Get(i, c) + delta;
                ysOut.Set(i, c, updatedValue);
            }
        }
    }
}

// void ForestPredictMaxYs(const Forest& forest, BufferCollection& data, const int numberOfindices, Int32VectorBuffer& maxYsOut)
// {
//     //Todo: implement
// }

int nextChild( const Tree& tree, int nodeId, BufferCollection& data, const int index );

int walkTree( const Tree& tree, int nodeId, BufferCollection& data, const int index, int& treeDepthOut )
{
    const int childNodeId = nextChild( tree, nodeId, data, index);
    if(childNodeId == NULL_CHILD)
    {
       return nodeId;
    }
    treeDepthOut++;
    return walkTree(tree, childNodeId, data, index, treeDepthOut);
}

int nextChild( const Tree& tree, int nodeId, BufferCollection& data, const int index )
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
            ASSERT( data.HasFloat32MatrixBuffer(X_FLOAT_DATA) )
            const Float32MatrixBuffer& xs = data.GetFloat32MatrixBuffer(X_FLOAT_DATA);
            int component = tree.mIntFeatureParams.Get(nodeId, 1);
            testResult = xs.Get(index, component) > threshold;
            break;
        }
        case VEC_FEATURE_PROJECTION:
        {
            ASSERT( data.HasFloat32MatrixBuffer(X_FLOAT_DATA) )
            const Float32MatrixBuffer& xs = data.GetFloat32MatrixBuffer(X_FLOAT_DATA);

            float projectionValue = 0.0f;
            const int numberOfComponentsInProjection = tree.mIntFeatureParams.Get(nodeId, 1);
            for(int p=2; p<numberOfComponentsInProjection+2; p++)
            {
                const int componentId = tree.mIntFeatureParams.Get(nodeId, p);
                const float componentProjection = tree.mFloatFeatureParams.Get(nodeId, p);
                projectionValue += xs.Get(index, componentId) * componentProjection;
            }
            testResult = projectionValue > threshold;
            break;
        }
        case IMG_FEATURE_DEPTH_DELTA:
        {
            ASSERT( data.HasFloat32Tensor3Buffer(DEPTH_IMAGES) )
            const Float32Tensor3Buffer& depths = data.GetFloat32Tensor3Buffer(DEPTH_IMAGES);
            ASSERT( data.HasInt32MatrixBuffer(PIXEL_INDICES) )
            const Int32MatrixBuffer& pixelIndices = data.GetInt32MatrixBuffer(PIXEL_INDICES);
            testResult = pixelDepthDeltaTest(depths, pixelIndices, index, tree.mFloatFeatureParams, nodeId );
            break;
        }
    }
    const int childDirection = testResult ? 0 : 1;
    const int childNodeId = tree.mPath.Get(nodeId, childDirection);
    return childNodeId;
}


