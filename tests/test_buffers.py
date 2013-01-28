import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers


class TestBuffers(unittest.TestCase):

    def test_img_float_buffer(self):
        A = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.float32 )
        a = buffers.Float32Tensor3( A )
        B = np.zeros((2,2,3), dtype=np.float32)
        a.AsNumpy3dFloat32(B)
        self.assertTrue((A == B).all())
        self.assertEqual(a.Get(1,0,2), 4)
        a.Set(1,0,2, 22)
        self.assertEqual(a.Get(1,0,2), 22)

    def test_img_int_buffer(self):
        A = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.int32 )
        a = buffers.Int32Tensor3( A )
        B = np.zeros((2,2,3), dtype=np.int32)
        a.AsNumpy3dInt32(B)
        self.assertTrue((A == B).all())
        self.assertEqual(a.Get(1,0,2), 4)
        a.Set(1,0,2, 22)
        self.assertEqual(a.Get(1,0,2), 22)

    def test_matrix_2d_float_buffer(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.float32 )
        a = buffers.Float32Matrix( A )
        B = np.zeros((2,3), dtype=np.float32)
        a.AsNumpy2dFloat32(B)
        self.assertTrue((A == B).all())
        self.assertEqual(a.Get(1,2), 5)
        a.Set(0,2,22)
        self.assertEqual(a.Get(0,2), 22)

    def test_matrix_2d_int_buffer(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix( A )
        B = np.zeros((2,3), dtype=np.int32)
        a.AsNumpy2dInt32(B)
        self.assertTrue((A == B).all())
        self.assertEqual(a.Get(1,2), 5)
        a.Set(0,2,22)
        self.assertEqual(a.Get(0,2), 22)

    def test_matrix_1d_float_buffer(self):
        A = np.array([3,21,1], dtype=np.float32 )
        a = buffers.vecBufferFloat( A )
        B = np.zeros((3,1), dtype=np.float32)
        a.AsNumpy2dFloat32(B)
        self.assertTrue((A == B.flatten()).all())
        self.assertEqual(a.Get(2,0), 1)
        a.Set(2,0,22)
        self.assertEqual(a.Get(2,0), 22)

    def test_matrix_1d_int_buffer(self):
        A = np.array([3,21,1], dtype=np.int32 )
        a = buffers.vecBufferInt( A )
        B = np.zeros((3,1), dtype=np.int32)
        a.AsNumpy2dInt32(B)
        self.assertTrue((A == B.flatten()).all())
        self.assertEqual(a.Get(2,0), 1)
        a.Set(2,0,22)
        self.assertEqual(a.Get(2,0), 22)

    def test_img_float_buffer_dim_exception(self):
        A = np.array([[[3,21,1],[22,1,5]],[[2,2,2],[7,7,7]]], dtype=np.float32 )
        a = buffers.Float32Tensor3( A )
        B = np.zeros((2,3,3), dtype=np.float32)
        with self.assertRaises(TypeError):
            a.AsNumpy3dFloat32(B)

    def test_img_int_buffer_dim_exception(self):
        A = np.array([[[3,21,1],[22,1,5]],[[2,2,2],[7,7,7]]], dtype=np.int32 )
        a = buffers.Int32Tensor3( A )
        B = np.zeros((1,2,3), dtype=np.int32)
        with self.assertRaises(TypeError):
            a.AsNumpy3dInt32(B)

    def test_matrix_float_buffer_dim_exception(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.float32 )
        a = buffers.Float32Matrix( A )
        B = np.zeros((1,3), dtype=np.float32)
        with self.assertRaises(TypeError):
            a.AsNumpy2dFloat32(B)

    def test_matrix_int_buffer_dim_exception(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix( A )
        B = np.zeros((2,1), dtype=np.int32)
        with self.assertRaises(TypeError):
            a.AsNumpy2dInt32(B)

    def test_img_float_buffer_out_of_range_exception(self):
        A = np.array([[[3,21,1],[22,1,5]],[[2,2,2],[7,7,7]]], dtype=np.float32 )
        a = buffers.Float32Tensor3( A )
        with self.assertRaises(IndexError):
            a.Get(2,0,0)
        with self.assertRaises(IndexError):
            a.Get(-1,0,0)
        with self.assertRaises(IndexError):
            a.Get(0,2,0)
        with self.assertRaises(IndexError):
            a.Get(0,-1,0)
        with self.assertRaises(IndexError):
            a.Get(0,0,3)
        with self.assertRaises(IndexError):
            a.Get(0,0,-1)

    def test_img_int_buffer_out_of_range_exception(self):
        A = np.array([[[3,21,1],[22,1,5]],[[2,2,2],[7,7,7]]], dtype=np.int32 )
        a = buffers.Int32Tensor3( A )
        with self.assertRaises(IndexError):
            a.Get(2,0,0)
        with self.assertRaises(IndexError):
            a.Get(-1,0,0)
        with self.assertRaises(IndexError):
            a.Get(0,2,0)
        with self.assertRaises(IndexError):
            a.Get(0,-1,0)
        with self.assertRaises(IndexError):
            a.Get(0,0,3)
        with self.assertRaises(IndexError):
            a.Get(0,0,-1)

    def test_matrix_float_buffer_out_of_range_exception(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.float32 )
        a = buffers.Float32Matrix( A )
        with self.assertRaises(IndexError):
            a.Get(2,0)
        with self.assertRaises(IndexError):
            a.Get(-1,0)
        with self.assertRaises(IndexError):
            a.Get(0,3)
        with self.assertRaises(IndexError):
            a.Get(0,-1)

    def test_matrix_int_buffer_out_of_range_exception(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix( A )
        with self.assertRaises(IndexError):
            a.Get(2,0)
        with self.assertRaises(IndexError):
            a.Get(-1,0)
        with self.assertRaises(IndexError):
            a.Get(0,3)
        with self.assertRaises(IndexError):
            a.Get(0,-1)

    # def test_buffer_shared_memory_copy(self):
    #     data_float = np.arange(1000000).reshape(1000,1000)
    #     data_int = np.array( data_float, dtype=np.int32)

    #     matrix_buffer_float = buffers.Float64Matrix( data_float )
    #     matrix_buffer_int = buffers.Int32Matrix( data_int )
    #     img_buffer_float = buffers.Float32Tensor3Buffer64( data_float )
    #     img_buffer_int = buffers.Float32Tensor3Buffer64( data_int )

    #     # Check that changing a memory copy changes the original
    #     copy = matrix_buffer_float.SharedMemoryCopy()
    #     copy.Set(100, 100, -22)
    #     self.assertEqual( matrix_buffer_float.Get(100,100), -22 )

    #     copy = matrix_buffer_int.SharedMemoryCopy()
    #     copy.Set(100, 100, -22)
    #     self.assertEqual( matrix_buffer_int.Get(100,100), -22 )

    #     copy = img_buffer_float.SharedMemoryCopy()
    #     copy.Set(0, 100, 100, -22)
    #     self.assertEqual( img_buffer_float.Get(0, 100,100), -22 )

    #     copy = img_buffer_int.SharedMemoryCopy()
    #     copy.Set(0, 100, 100, -22)
    #     self.assertEqual( img_buffer_int.Get(0, 100,100), -22 )

    #     # The loop below just checks than very little memory is used per copy
    #     list_of_ref = []
    #     for i in range(10000):
    #         list_of_ref.append( matrix_buffer_float.SharedMemoryCopy() )
    #         list_of_ref.append( matrix_buffer_int.SharedMemoryCopy() )
    #         list_of_ref.append( img_buffer_float.SharedMemoryCopy() )
    #         list_of_ref.append( img_buffer_int.SharedMemoryCopy() )

    def test_matrix_buffer_minmax(self):
        A_float = np.array([[3,21,-1],[22,33,5]], dtype=np.float32 )
        a_float = buffers.Float32Matrix( A_float )
        self.assertEqual( a_float.GetMax(), 33)
        self.assertEqual( a_float.GetMin(), -1)

        A_int = np.array([[3,21,-5],[22,33,5]], dtype=np.int32 )
        a_int = buffers.Int32Matrix( A_int )
        self.assertEqual( a_int.GetMax(), 33)
        self.assertEqual( a_int.GetMin(), -5)

    def test_matrix_buffer_zero(self):
        A_nonzero = np.array([[3,21,1],[22,33,5]], dtype=np.float32 )
        a = buffers.Float32Matrix( A_nonzero )
        result = np.zeros((2,3), dtype=np.float32)
        a.AsNumpy2dFloat32(result)
        self.assertTrue((A_nonzero == result).all())
        a.Zero()
        a.AsNumpy2dFloat32(result)
        zeros = np.zeros((2,3), dtype=np.float32)
        self.assertTrue((zeros == result).all())

        A_nonzero = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix( A_nonzero )
        result = np.zeros((2,3), dtype=np.int32)
        a.AsNumpy2dInt32(result)
        self.assertTrue((A_nonzero == result).all())
        a.Zero()
        a.AsNumpy2dInt32(result)
        zeros = np.zeros((2,3), dtype=np.int32)
        self.assertTrue((zeros == result).all())



if __name__ == '__main__':
    unittest.main()