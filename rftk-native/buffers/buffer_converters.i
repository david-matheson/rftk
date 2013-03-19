/* Converters */
%pythoncode %{
import numpy as np
import scipy.sparse
import buffers as buffers

def as_buffer( np_array ):
    if scipy.sparse.issparse(np_array):
        return as_sparse_matrix(np_array)
    elif np_array.ndim == 1:
        return as_vector_buffer(np_array)
    elif np_array.ndim == 2:
        return as_matrix_buffer(np_array)
    elif np_array.ndim == 3:
        return as_tensor_buffer(np_array)
    else:
        raise Exception('as_buffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_vector_buffer( np_array ):
    type_string = np_array.dtype.name.title()
    function_name = '%s%s' % (type_string, 'Vector')
    if hasattr(buffers, function_name):
        function = getattr(buffers, function_name)
        return function(np_array)
    else:
        raise Exception('as_vector_buffer failed because %s does not exist' % function_name)

def as_matrix_buffer( np_array ):
    type_string = np_array.dtype.name.title()
    function_name = '%s%s%d' % (type_string, 'Matrix', np_array.ndim)
    if hasattr(buffers, function_name):
        function = getattr(buffers, function_name)
        return function(np_array)
    else:
        raise Exception('as_matrix_buffer failed because %s does not exist' % function_name)

def as_sparse_matrix( sparse_matrix ):
    type_string = np.sparse_matrix.dtype.name.title()
    function_name = '%s%s%d' % (type_string, 'SparseMatrix', sparse_matrix.ndim)
    if hasattr(buffers, function_name):
        function = getattr(buffers, function_name)
        return function(np_array)
    else:
        raise Exception('as_sparse_matrix failed because %s does not exist' % function_name)


def as_tensor_buffer( np_array ):
    type_string = np_array.dtype.name.title()
    function_name = '%s%s%d' % (type_string, 'Tensor', np_array.ndim)
    if hasattr(buffers, function_name):
        function = getattr(buffers, function_name)
        return function(np_array)
    else:
        raise Exception('as_tensor_buffer failed because %s does not exist' % function_name)

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


def is_buffer( buffer ):
    isFloatTensor3Buffer = isinstance(buffer, buffers.Float32Tensor3Buffer) or isinstance(buffer, buffers.Float64Tensor3Buffer)
    isIntTensor3Buffer = isinstance(buffer, buffers.Int32Tensor3Buffer) or isinstance(buffer, buffers.Int64Tensor3Buffer)
    isFloatMatrixBuffer = isinstance(buffer, buffers.Float32MatrixBuffer) or isinstance(buffer, buffers.Float64MatrixBuffer)
    isIntMatrixBuffer = isinstance(buffer, buffers.Int32MatrixBuffer) or isinstance(buffer, buffers.Int64MatrixBuffer)
    isFloatVectorBuffer = isinstance(buffer, buffers.Float32VectorBuffer) or isinstance(buffer, buffers.Float64VectorBuffer)
    isIntVectorBuffer = isinstance(buffer, buffers.Int32VectorBuffer) or isinstance(buffer, buffers.Int64VectorBuffer)
    return isFloatTensor3Buffer or isIntTensor3Buffer or isFloatMatrixBuffer or isIntMatrixBuffer or isFloatVectorBuffer or isIntVectorBuffer
%}