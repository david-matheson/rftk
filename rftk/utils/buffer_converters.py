import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers

def as_img_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.imgBufferInt( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 3:
        return buffers.imgsBufferInt( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.imgBufferFloat( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 3:
        return buffers.imgsBufferFloat( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.imgBufferInt64( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 3:
        return buffers.imgsBufferInt64( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.imgBufferFloat64( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 3:
        return buffers.imgsBufferFloat64( np_array )
    else:
        raise Exception('asImgBuffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_matrix_buffer( np_array ):
    if np_array.dtype == np.int32 and np_array.ndim == 1:
        return buffers.vecBufferInt( np_array )
    elif np_array.dtype == np.int32 and np_array.ndim == 2:
        return buffers.matrixBufferInt( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 1:
        return buffers.vecBufferFloat( np_array )
    elif np_array.dtype == np.float32 and np_array.ndim == 2:
        return buffers.matrixBufferFloat( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 1:
        return buffers.vecBufferInt64( np_array )
    elif np_array.dtype == np.int64 and np_array.ndim == 2:
        return buffers.matrixBufferInt64( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 1:
        return buffers.vecBufferFloat64( np_array )
    elif np_array.dtype == np.float64 and np_array.ndim == 2:
        return buffers.matrixBufferFloat64( np_array )
    else:
        raise Exception('asMatrixBuffer unknown type and ndim', np_array.dtype, np_array.ndim())

def as_numpy_array( buffer, flatten=False ):
    isImgBufferFloat = isinstance(buffer, buffers.ImgBufferFloat)
    isImgBufferInt = isinstance(buffer, buffers.ImgBufferInt)
    isMatrixBufferFloat = isinstance(buffer, buffers.MatrixBufferFloat)
    isMatrixBufferInt = isinstance(buffer, buffers.MatrixBufferInt)

    assert(isImgBufferFloat or isImgBufferInt or isMatrixBufferFloat or isMatrixBufferInt)

    if isImgBufferFloat or isMatrixBufferFloat:
        buffer_type = np.float32
    else:
        buffer_type = np.int32

    if isImgBufferFloat or isImgBufferInt:
        result = np.zeros((buffer.GetNumberOfImgs(), buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        buffer.AsNumpy(result)
        if buffer.GetNumberOfImgs() == 1 and flatten:
            result = result.reshape(buffer.GetM(), buffer.GetN)

    elif isMatrixBufferFloat or isMatrixBufferInt:
        result = np.zeros((buffer.GetM(), buffer.GetN()), dtype=buffer_type)
        buffer.AsNumpy(result)
        if buffer.GetN() == 1 and flatten:
            result = result.flatten()

    return result






