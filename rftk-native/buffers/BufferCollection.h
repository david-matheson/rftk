#pragma once

#include <string>
#include <map>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"

//Using #define for compatibility with swig
#define X_FLOAT_DATA    "X_Float"
#define SAMPLE_WEIGHTS  "SampleWeights"
#define CLASS_LABELS    "ClassLabels"
#define FEATURE_VALUES  "Feature_Values"
#define HISTOGRAM_LEFT            "Histogram_Left"
#define HISTOGRAM_RIGHT           "Histogram_Right"
#define THRESHOLDS                "Thresholds"
#define THRESHOLD_COUNTS          "ThresholdCounts"
#define IMPURITY_HISTOGRAM_LEFT   "Impurity_Histogram_Left"
#define IMPURITY_HISTOGRAM_RIGHT  "Impurity_Histogram_Right"
#define YS_HISTOGRAM_LEFT         "Ys_Histogram_Left"
#define YS_HISTOGRAM_RIGHT        "Ys_Histogram_Right"

#define PIXEL_INDICES   "PixelIndices"
#define DEPTH_IMAGES    "DepthImages"
#define OFFSET_SCALES   "OffsetScales"

class BufferCollection
{
public:
    BufferCollection();

    bool HasFloat32VectorBuffer(std::string name) const;
    void AddFloat32VectorBuffer(std::string name, const Float32VectorBuffer& data );
    void AppendFloat32VectorBuffer(std::string name, const Float32VectorBuffer& data );
    const Float32VectorBuffer& GetFloat32VectorBuffer(const std::string& name) const;
    Float32VectorBuffer& GetFloat32VectorBuffer(const std::string& name);

    bool HasInt32VectorBuffer(std::string name) const;
    void AddInt32VectorBuffer(std::string name, const Int32VectorBuffer& data );
    void AppendInt32VectorBuffer(std::string name, const Int32VectorBuffer& data );
    const Int32VectorBuffer& GetInt32VectorBuffer(const std::string& name) const;
    Int32VectorBuffer& GetInt32VectorBuffer(const std::string& name);

    bool HasFloat32MatrixBuffer(std::string name) const;
    void AddFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data );
    void AppendVerticalFloat32MatrixBuffer(std::string name, const Float32MatrixBuffer& data );
    const Float32MatrixBuffer& GetFloat32MatrixBuffer(const std::string& name) const;
    Float32MatrixBuffer& GetFloat32MatrixBuffer(const std::string& name);

    bool HasInt32MatrixBuffer(std::string name) const;
    void AddInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data );
    void AppendVerticalInt32MatrixBuffer(std::string name, const Int32MatrixBuffer& data );
    const Int32MatrixBuffer& GetInt32MatrixBuffer(const std::string& name) const;
    Int32MatrixBuffer& GetInt32MatrixBuffer(const std::string& name);

    bool HasFloat32Tensor3Buffer(std::string name) const;
    void AddFloat32Tensor3Buffer(std::string name, const Float32Tensor3Buffer& data );
    const Float32Tensor3Buffer& GetFloat32Tensor3Buffer(std::string name) const;
    Float32Tensor3Buffer& GetFloat32Tensor3Buffer(std::string name);

    bool HasInt32Tensor3Buffer(std::string name) const;
    void AddInt32Tensor3Buffer(std::string name, const Int32Tensor3Buffer& data ) ;
    const Int32Tensor3Buffer& GetInt32Tensor3Buffer(std::string name) const;
    Int32Tensor3Buffer& GetInt32Tensor3Buffer(std::string name);

private:
    std::map<std::string, Float32VectorBuffer> mFloat32VectorBuffers;
    std::map<std::string, Int32VectorBuffer> mInt32VectorBuffers;
    std::map<std::string, Float32MatrixBuffer> mFloat32MatrixBuffers;
    std::map<std::string, Int32MatrixBuffer> mInt32MatrixBuffers;
    std::map<std::string, Float32Tensor3Buffer> mFloat32Tensor3Buffers;
    std::map<std::string, Int32Tensor3Buffer> mInt32Tensor3Buffers;
};