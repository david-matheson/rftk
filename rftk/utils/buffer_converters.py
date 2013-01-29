import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers

def as_img_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.Int32Tensor2( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 3:
        return buffers.Int32Tensor3( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.Float32Tensor2( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 3:
        return buffers.Float32Tensor3( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.Int32Tensor2( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 3:
        return buffers.Int32Tensor3( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.Float32Tensor2( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 3:
        return buffers.Float32Tensor3( np_array )
    else:
        raise Exception('asImgBuffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_matrix_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 1:
        return buffers.vecBufferInt( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.Int32Matrix( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 1:
        return buffers.vecBufferFloat( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.Float32Matrix( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 1:
        return buffers.vecBufferInt64( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.Int64Matrix( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 1:
        return buffers.vecBufferFloat64( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.Float64Matrix( np_array )
    else:
        raise Exception('asMatrixBuffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_numpy_array( buffer, flatten=False ):
    isFloat32Tensor3Buffer = isinstance(buffer, buffers.Float32Tensor3Buffer)
    isInt32Tensor3Buffer = isinstance(buffer, buffers.Int32Tensor3Buffer)
    isFloatMatrixBuffer = isinstance(buffer, buffers.Float32MatrixBuffer) or isinstance(buffer, buffers.Float64MatrixBuffer)
    isIntMatrixBuffer = isinstance(buffer, buffers.Int32MatrixBuffer) or isinstance(buffer, buffers.Int64MatrixBuffer)
    isFloatVectorBuffer = isinstance(buffer, buffers.Float32VectorBuffer) or isinstance(buffer, buffers.Float64VectorBuffer)
    isIntVectorBuffer = isinstance(buffer, buffers.Int32VectorBuffer) or isinstance(buffer, buffers.Int64VectorBuffer)

    assert(isFloat32Tensor3Buffer or isInt32Tensor3Buffer 
        or isFloatMatrixBuffer or isIntMatrixBuffer
        or isFloatVectorBuffer or isIntVectorBuffer)

    if isFloat32Tensor3Buffer or isFloatMatrixBuffer or isFloatVectorBuffer:
        buffer_type = np.float32
    else:
        buffer_type = np.int32

    if isFloat32Tensor3Buffer or isInt32Tensor3Buffer:
        result = np.zeros((buffer.GetL(), buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        if isFloat32Tensor3Buffer:
            buffer.AsNumpy3dFloat32(result)
        if isInt32Tensor3Buffer:
            buffer.AsNumpy3dInt32(result)
        if buffer.GetL() == 1 and flatten:
            result = result.reshape(buffer.GetM(), buffer.GetN)

    elif isFloatMatrixBuffer or isIntMatrixBuffer:
        result = np.zeros((buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        if isFloatMatrixBuffer:
            buffer.AsNumpy2dFloat32(result)
        if isIntMatrixBuffer:
            buffer.AsNumpy2dInt32(result)
        if buffer.GetN() == 1 and flatten:
            result = result.flatten()

    elif isFloatVectorBuffer or isIntVectorBuffer:
        result = np.zeros(buffer.GetN(), dtype=buffer_type)
        if isFloatVectorBuffer:
            buffer.AsNumpy1dFloat32(result)
        if isIntVectorBuffer:
            buffer.AsNumpy1dInt32(result)

    return result






