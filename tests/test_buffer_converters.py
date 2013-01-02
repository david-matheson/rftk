import unittest as unittest
import numpy as np
import rftk.assert_util
import rftk.buffers as buffers
import rftk.utils.buffer_converters as buffer_converters


class TestBufferConverters(unittest.TestCase):

    def convert_img_buffer_both_directions_helper(self, X, buffer_type):
        buf = buffer_converters.as_img_buffer(X)
        assert isinstance(buf, buffer_type)
        X_back = buffer_converters.as_numpy_array(buf)
        self.assertTrue((X == X_back).all())

    def img_buffer_flatten_helper(self, X, buffer_type):  
        m,n = X.shape       
        img_buffer = buffer_converters.as_img_buffer(X)
        assert isinstance(img_buffer, buffer_type)
        X_back = buffer_converters.as_numpy_array(img_buffer)
        self.assertTrue((X.reshape(1,m,n) == X_back).all())
        X_back = buffer_converters.as_numpy_array(buffer=img_buffer, flatten=True)
        self.assertTrue((X == X_back).all())

    def convert_matrix_buffer_both_directions_helper(self, X, buffer_type):
        buf = buffer_converters.as_matrix_buffer(X)
        assert isinstance(buf, buffer_type)
        X_back = buffer_converters.as_numpy_array(buf)
        self.assertTrue((X == X_back).all())

    def matrix_buffer_flatten_helper(self, X, buffer_type):     
        n = len(X.shape)   
        matrix_buffer = buffer_converters.as_matrix_buffer(X)
        assert isinstance(matrix_buffer, buffer_type)
        X_back = buffer_converters.as_numpy_array(matrix_buffer)
        self.assertTrue((X == X_back.flatten()).all())
        X_back = buffer_converters.as_numpy_array(buffer=matrix_buffer, flatten=True)
        self.assertTrue((X == X_back).all())

    def test_img_buffer(self):
        X = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.int32 )
        self.convert_img_buffer_both_directions_helper(X=X, buffer_type=buffers.ImgBufferInt)
        X = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.int )
        self.convert_img_buffer_both_directions_helper(X=X, buffer_type=buffers.ImgBufferInt)
        X = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.float32 )
        self.convert_img_buffer_both_directions_helper(X=X, buffer_type=buffers.ImgBufferFloat)
        X = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.float )
        self.convert_img_buffer_both_directions_helper(X=X, buffer_type=buffers.ImgBufferFloat)
        

    def test_img_buffer_flatten(self):
        X = np.array([[3,21,1],[22,1,5]], dtype=np.int32 )
        self.img_buffer_flatten_helper(X=X, buffer_type=buffers.ImgBufferInt)
        X_64 = np.array([[3,21,1],[22,1,5]], dtype=np.int )
        self.img_buffer_flatten_helper(X=X_64, buffer_type=buffers.ImgBufferInt)
        X = np.array([[3,21,1],[22,1,5]], dtype=np.float32 )
        self.img_buffer_flatten_helper(X=X, buffer_type=buffers.ImgBufferFloat)
        X_64 = np.array([[3,21,1],[22,1,5]], dtype=np.float )
        self.img_buffer_flatten_helper(X=X_64, buffer_type=buffers.ImgBufferFloat)

    def test_matrix_buffer(self):
        X = np.array([[3,21,1],[22,1,5]], dtype=np.int32 )
        self.convert_matrix_buffer_both_directions_helper(X=X, buffer_type=buffers.MatrixBufferInt)
        X = np.array([[3,21,1],[22,1,5]], dtype=np.int )
        self.convert_matrix_buffer_both_directions_helper(X=X, buffer_type=buffers.MatrixBufferInt)
        X = np.array([[3,21,1],[22,1,5]], dtype=np.float32 )
        self.convert_matrix_buffer_both_directions_helper(X=X, buffer_type=buffers.MatrixBufferFloat)
        X = np.array([[3,21,1],[22,1,5]], dtype=np.float )
        self.convert_matrix_buffer_both_directions_helper(X=X, buffer_type=buffers.MatrixBufferFloat)
        

    def test_matrix_buffer_flatten(self):
        X = np.array([22,1,5], dtype=np.int32 )
        self.matrix_buffer_flatten_helper(X=X, buffer_type=buffers.MatrixBufferInt)
        X_64 = np.array([22,1,5], dtype=np.int )
        self.matrix_buffer_flatten_helper(X=X_64, buffer_type=buffers.MatrixBufferInt)
        X = np.array([22,1,5], dtype=np.float32 )
        self.matrix_buffer_flatten_helper(X=X, buffer_type=buffers.MatrixBufferFloat)
        X_64 = np.array([22,1,5], dtype=np.float )
        self.matrix_buffer_flatten_helper(X=X_64, buffer_type=buffers.MatrixBufferFloat)





if __name__ == '__main__':
    unittest.main()
