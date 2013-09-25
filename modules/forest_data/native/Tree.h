#pragma once

#include <boost/shared_ptr.hpp>

#include <VectorBuffer.h>
#include <MatrixBuffer.h>
#include <BufferCollection.h>

const int NULL_CHILD = -1;

class ForestStats;

class TreeData
{
public:
    TreeData();     //default for stl vector

    TreeData(   const MatrixBufferTemplate<int>& path,
            const MatrixBufferTemplate<int>& intFeatureParams,
            const MatrixBufferTemplate<float>& floatFeatureParams,
            const VectorBufferTemplate<int> & depths,
            const VectorBufferTemplate<float>& counts,
            const MatrixBufferTemplate<float>& ys );

    TreeData( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim );
    TreeData(const TreeData& tree);
    ~TreeData();

    int NextNodeIndex();
    void Compact();

    MatrixBufferTemplate<int> mPath;
    MatrixBufferTemplate<int> mIntFeatureParams;
    MatrixBufferTemplate<float> mFloatFeatureParams;
    VectorBufferTemplate<float> mCounts;
    VectorBufferTemplate<int> mDepths;
    MatrixBufferTemplate<float> mYs;

    BufferCollection mExtraInfo; 

    int mLastNodeIndex;
    bool mValid;
};

class Tree
{
public:
    Tree();     //default for stl vector

    Tree(   const MatrixBufferTemplate<int>& path,
            const MatrixBufferTemplate<int>& intFeatureParams,
            const MatrixBufferTemplate<float>& floatFeatureParams,
            const VectorBufferTemplate<int> & depths,
            const VectorBufferTemplate<float>& counts,
            const MatrixBufferTemplate<float>& ys );

    Tree( int initalNumberNodes, int maxIntParamsDim, int maxFloatParamsDim, int maxYsDim );

    ~Tree();

    void GatherStats(ForestStats& stats) const;
    int NextNodeIndex();
    void Compact();

    MatrixBufferTemplate<int>& GetPath();
    MatrixBufferTemplate<int>& GetIntFeatureParams();
    MatrixBufferTemplate<float>& GetFloatFeatureParams();
    VectorBufferTemplate<float>& GetCounts();
    VectorBufferTemplate<int>& GetDepths();
    MatrixBufferTemplate<float>& GetYs();
    BufferCollection& GetExtraInfo(); 

    const MatrixBufferTemplate<int>& GetPath() const;
    const MatrixBufferTemplate<int>& GetIntFeatureParams() const;
    const MatrixBufferTemplate<float>& GetFloatFeatureParams() const;
    const VectorBufferTemplate<float>& GetCounts() const;
    const VectorBufferTemplate<int>& GetDepths() const;
    const MatrixBufferTemplate<float>& GetYs() const;
    const BufferCollection& GetExtraInfo() const; 

private:
    boost::shared_ptr<TreeData> mTreeData;
};

