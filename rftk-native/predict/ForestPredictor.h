#pragma once

#include "BufferCollection.h"

#include "Forest.h"


class ForestPredictor
{
public:
    ForestPredictor( const Forest& forest );

    void PredictLeafs(BufferCollection& data, const int numberOfindices, MatrixBufferInt& leafsOut);
    void PredictYs(BufferCollection& data, const int numberOfindices, MatrixBufferFloat& ysOut);

    Forest mForest;
};

void ForestPredictLeafs(const Forest& forest, BufferCollection& data, const int numberOfindices, MatrixBufferInt& leafsOut);
void ForestPredictYs(const Forest& forest, BufferCollection& data, const int numberOfindices, MatrixBufferFloat& ysOut);
int walkTree( const Tree& tree, int nodeId, BufferCollection& data, const int index, int& treeDepthOut );