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
    type_string = sparse_matrix.dtype.name.title()
    function_name = '%s%s' % (type_string, 'SparseMatrix')
    if hasattr(buffers, function_name):
        function = getattr(buffers, function_name)
        return function(sparse_matrix)
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
    isFloat32Tensor3Buffer = isinstance(buffer, buffers.Float32Tensor3Buffer)
    isFloat64Tensor3Buffer = isinstance(buffer, buffers.Float64Tensor3Buffer)
    isInt32Tensor3Buffer = isinstance(buffer, buffers.Int32Tensor3Buffer)
    isInt64Tensor3Buffer = isinstance(buffer, buffers.Int64Tensor3Buffer)
    isFloat32MatrixBuffer = isinstance(buffer, buffers.Float32MatrixBuffer)
    isFloat64MatrixBuffer = isinstance(buffer, buffers.Float64MatrixBuffer)
    isInt32MatrixBuffer = isinstance(buffer, buffers.Int32MatrixBuffer)
    isInt64MatrixBuffer = isinstance(buffer, buffers.Int64MatrixBuffer)
    isFloat32VectorBuffer = isinstance(buffer, buffers.Float32VectorBuffer)
    isFloat64VectorBuffer = isinstance(buffer, buffers.Float64VectorBuffer)
    isInt32VectorBuffer = isinstance(buffer, buffers.Int32VectorBuffer)
    isInt64VectorBuffer = isinstance(buffer, buffers.Int64VectorBuffer)

    assert(isFloat32Tensor3Buffer or isFloat64Tensor3Buffer or isInt32Tensor3Buffer or isInt64Tensor3Buffer
        or isFloat32MatrixBuffer or isFloat64MatrixBuffer or isInt32MatrixBuffer or isInt64MatrixBuffer
        or isFloat32VectorBuffer or isFloat64VectorBuffer or isInt32VectorBuffer or isInt64VectorBuffer)

    if isFloat32Tensor3Buffer or isFloat32MatrixBuffer or isFloat32VectorBuffer:
        buffer_type = np.float32
    elif isFloat64Tensor3Buffer or isFloat64MatrixBuffer or isFloat64VectorBuffer:
        buffer_type = np.float64
    elif isInt32Tensor3Buffer or isInt32MatrixBuffer or isInt32VectorBuffer:
        buffer_type = np.int32
    elif isInt64Tensor3Buffer or isInt64MatrixBuffer or isInt64VectorBuffer:
        buffer_type = np.int64

    if isFloat32Tensor3Buffer or isFloat64Tensor3Buffer or isInt32Tensor3Buffer or isInt64Tensor3Buffer:
        result = np.zeros((buffer.GetL(), buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        function_name = 'AsNumpy3d%s' % result.dtype.name.title()
        function = getattr(buffer, function_name)
        function(result)

        if buffer.GetL() == 1 and flatten:
            result = result.reshape(buffer.GetM(), buffer.GetN)

    elif isFloat32MatrixBuffer or isFloat64MatrixBuffer or isInt32MatrixBuffer or isInt64MatrixBuffer:
        result = np.zeros((buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        function_name = 'AsNumpy2d%s' % result.dtype.name.title()
        function = getattr(buffer, function_name)
        function(result)

        if buffer.GetN() == 1 and flatten:
            result = result.flatten()

    elif isFloat32VectorBuffer or isFloat64VectorBuffer or isInt32VectorBuffer or isInt64VectorBuffer:
        result = np.zeros(buffer.GetN(), dtype=buffer_type)
        function_name = 'AsNumpy1d%s' % result.dtype.name.title()
        function = getattr(buffer, function_name)
        function(result)

    return result


def is_buffer( buffer ):
    isFloatTensor3Buffer = isinstance(buffer, buffers.Float32Tensor3Buffer) or isinstance(buffer, buffers.Float64Tensor3Buffer)
    isIntTensor3Buffer = isinstance(buffer, buffers.Int32Tensor3Buffer) or isinstance(buffer, buffers.Int64Tensor3Buffer)
    isFloatMatrixBuffer = isinstance(buffer, buffers.Float32MatrixBuffer) or isinstance(buffer, buffers.Float64MatrixBuffer)
    isIntMatrixBuffer = isinstance(buffer, buffers.Int32MatrixBuffer) or isinstance(buffer, buffers.Int64MatrixBuffer)
    isFloatVectorBuffer = isinstance(buffer, buffers.Float32VectorBuffer) or isinstance(buffer, buffers.Float64VectorBuffer)
    isIntVectorBuffer = isinstance(buffer, buffers.Int32VectorBuffer) or isinstance(buffer, buffers.Int64VectorBuffer)
    return isFloatTensor3Buffer or isIntTensor3Buffer or isFloatMatrixBuffer or isIntMatrixBuffer or isFloatVectorBuffer or isIntVectorBuffer