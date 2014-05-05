#pragma once

#include "Tree.h"

class ForestStats
{
public:
    ForestStats();
    void ProcessLeaf(const Tree& tree, int nodeId);
    float GetAverageDepth() const;
    float GetAverageEstimatorPoints() const;
    float GetAverageError() const;
    void Print() const;

    int mNumberOfLeafNodes;
    int mMinDepth;
    int mMaxDepth;
    int mTotalDepth;
    int mMinEstimatorPoints;
    int mMaxEstimatorPoints;
    int mTotalEstimatorPoints;
    float mMinError;
    float mMaxError;
    float mTotalError;
};