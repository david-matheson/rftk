#pragma once

#include <string>

#include <MatrixBuffer.h>
#include <BufferCollection.h>

#include "BestSplitI.h"


class ClassInfoGainHistogramsBestSplit : public BestSplitI {
public:
    ClassInfoGainHistogramsBestSplit( int numberOfClasses,
                                      const std::string& leftImpurityHistrogramBufferName,
                                      const std::string& rightImpurityHistrogramBufferName,
                                      const std::string& leftYsHistogramBufferName,
                                      const std::string& rightYsHistrogramBufferName);
    virtual ~ClassInfoGainHistogramsBestSplit();

    virtual BestSplitI* Clone() const;

    virtual int GetYDim() const;

    virtual void BestSplits( const BufferCollection& data,
                            Float32VectorBuffer& impurityOut,
                            Float32VectorBuffer& thresholdOut,
                            Float32MatrixBuffer& childCountsOut,
                            Float32MatrixBuffer& leftYsOut,
                            Float32MatrixBuffer& rightYsOut) const;

private:
  int mNumberOfClasses;
  std::string mLeftImpurityHistrogramBufferName;
  std::string mRightImpurityHistrogramBufferName;
  std::string mLeftYsHistogramBufferName;
  std::string mRightYsHistogramBufferName;
};