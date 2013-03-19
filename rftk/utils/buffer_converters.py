import numpy as np
import scipy.sparse

import rftk.native.assert_util
import rftk.native.buffers as buffers

def as_vector_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 1:
        return buffers.Int32Vector( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 1:
        return buffers.Float32Vector( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 1:
        return buffers.Int64Vector( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 1:
        return buffers.Float64Vector( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.Int32Vector( np_array.flatten() )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.Float32Vector( np_array.flatten() )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.Int64Vector( np_array.flatten() )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.Float64Vector( np_array.flatten() )
    else:
        raise Exception('as_vector_buffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_matrix_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 1:
        return buffers.Int32Matrix1( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.Int32Matrix2( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 1:
        return buffers.Float32Matrix1( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.Float32Matrix2( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 1:
        return buffers.Int64Matrix1( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.Int64Matrix2( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 1:
        return buffers.Float64Matrix1( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.Float64Matrix2( np_array )
    else:
        raise Exception('as_matrix_buffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_sparse_matrix( sparse_matrix ):
    if sparse_matrix.dtype == np.float32:
        return buffers.Float32SparseMatrix(sparse_matrix)
    elif sparse_matrix.dtype == np.float64:
        return buffers.Float64SparseMatrix(sparse_matrix)
    elif sparse_matrix.dtype == np.int32:
        return buffers.Int32SparseMatrix(sparse_matrix)
    elif sparse_matrix.dtype == np.int64:
        return buffers.Int64SparseMatrix(sparse_matrix)
    else:
        raise Exception('as_sparse_matrix unknown type', np_array.dtype)


def as_tensor_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.Int32Tensor2( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 3:
        return buffers.Int32Tensor3( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.Float32Tensor2( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 3:
        return buffers.Float32Tensor3( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.Int64Tensor2( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 3:
        return buffers.Int64Tensor3( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.Float64Tensor2( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 3:
        return buffers.Float64Tensor3( np_array )
    else:
        raise Exception('as_tensor_buffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_numpy_array( buffer, flatten=False ):
    isFloatTensor3Buffer = isinstance(buffer, buffers.Float32Tensor3Buffer) or isinstance(buffer, buffers.Float64Tensor3Buffer)
    isIntTensor3Buffer = isinstance(buffer, buffers.Int32Tensor3Buffer) or isinstance(buffer, buffers.Int64Tensor3Buffer)
    isFloatMatrixBuffer = isinstance(buffer, buffers.Float32MatrixBuffer) or isinstance(buffer, buffers.Float64MatrixBuffer)
    isIntMatrixBuffer = isinstance(buffer, buffers.Int32MatrixBuffer) or isinstance(buffer, buffers.Int64MatrixBuffer)
    isFloatVectorBuffer = isinstance(buffer, buffers.Float32VectorBuffer) or isinstance(buffer, buffers.Float64VectorBuffer)
    isIntVectorBuffer = isinstance(buffer, buffers.Int32VectorBuffer) or isinstance(buffer, buffers.Int64VectorBuffer)

    assert(isFloatTensor3Buffer or isIntTensor3Buffer
        or isFloatMatrixBuffer or isIntMatrixBuffer
        or isFloatVectorBuffer or isIntVectorBuffer)

    if isFloatTensor3Buffer or isFloatMatrixBuffer or isFloatVectorBuffer:
        buffer_type = np.float32
    else:
        buffer_type = np.int32

    if isFloatTensor3Buffer or isIntTensor3Buffer:
        result = np.zeros((buffer.GetL(), buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        if isFloatTensor3Buffer:
            buffer.AsNumpy3dFloat32(result)
        if isIntTensor3Buffer:
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






