#pragma once

#include "BufferCollection.h"

#include "Forest.h"


class ForestPredictor
{
public:
    ForestPredictor( const Forest& forest );

    void PredictLeafs(BufferCollection& data, const int numberOfindices, Int32MatrixBuffer& leafsOut);
    void PredictYs(BufferCollection& data, const int numberOfindices, Float32MatrixBuffer& ysOut);

    Forest mForest;
};

void ForestPredictLeafs(const Forest& forest, BufferCollection& data, const int numberOfindices, Int32MatrixBuffer& leafsOut);
void ForestPredictYs(const Forest& forest, BufferCollection& data, const int numberOfindices, Float32VectorBuffer& ysOut);
int walkTree( const Tree& tree, int nodeId, BufferCollection& data, const int index, int& treeDepthOut );