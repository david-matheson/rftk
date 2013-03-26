import unittest as unittest
import numpy as np

import pickle
import itertools

import rftk.buffers as buffers


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
        a = buffers.Float32Matrix2( A )
        B = np.zeros((2,3), dtype=np.float32)
        a.AsNumpy2dFloat32(B)
        self.assertTrue((A == B).all())
        self.assertEqual(a.Get(1,2), 5)
        a.Set(0,2,22)
        self.assertEqual(a.Get(0,2), 22)

    def test_matrix_2d_int_buffer(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix2( A )
        B = np.zeros((2,3), dtype=np.int32)
        a.AsNumpy2dInt32(B)
        self.assertTrue((A == B).all())
        self.assertEqual(a.Get(1,2), 5)
        a.Set(0,2,22)
        self.assertEqual(a.Get(0,2), 22)

    def test_matrix_1d_float_buffer(self):
        A = np.array([3,21,1], dtype=np.float32 )
        a = buffers.Float32Matrix1( A )
        B = np.zeros((3,1), dtype=np.float32)
        a.AsNumpy2dFloat32(B)
        self.assertTrue((A == B.flatten()).all())
        self.assertEqual(a.Get(2,0), 1)
        a.Set(2,0,22)
        self.assertEqual(a.Get(2,0), 22)

    def test_matrix_1d_int_buffer(self):
        A = np.array([3,21,1], dtype=np.int32 )
        a = buffers.Int32Matrix1( A )
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
        a = buffers.Float32Matrix2( A )
        B = np.zeros((1,3), dtype=np.float32)
        with self.assertRaises(TypeError):
            a.AsNumpy2dFloat32(B)

    def test_matrix_int_buffer_dim_exception(self):
        A = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix2( A )
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
        a = buffers.Float32Matrix2( A )
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
        a = buffers.Int32Matrix2( A )
        with self.assertRaises(IndexError):
            a.Get(2,0)
        with self.assertRaises(IndexError):
            a.Get(-1,0)
        with self.assertRaises(IndexError):
            a.Get(0,3)
        with self.assertRaises(IndexError):
            a.Get(0,-1)

    def test_matrix_buffer_minmax(self):
        A_float = np.array([[3,21,-1],[22,33,5]], dtype=np.float32 )
        a_float = buffers.Float32Matrix2( A_float )
        self.assertEqual( a_float.GetMax(), 33)
        self.assertEqual( a_float.GetMin(), -1)

        A_int = np.array([[3,21,-5],[22,33,5]], dtype=np.int32 )
        a_int = buffers.Int32Matrix2( A_int )
        self.assertEqual( a_int.GetMax(), 33)
        self.assertEqual( a_int.GetMin(), -5)

    def test_matrix_buffer_zero(self):
        A_nonzero = np.array([[3,21,1],[22,33,5]], dtype=np.float32 )
        a = buffers.Float32Matrix2( A_nonzero )
        result = np.zeros((2,3), dtype=np.float32)
        a.AsNumpy2dFloat32(result)
        self.assertTrue((A_nonzero == result).all())
        a.Zero()
        a.AsNumpy2dFloat32(result)
        zeros = np.zeros((2,3), dtype=np.float32)
        self.assertTrue((zeros == result).all())

        A_nonzero = np.array([[3,21,1],[22,33,5]], dtype=np.int32 )
        a = buffers.Int32Matrix2( A_nonzero )
        result = np.zeros((2,3), dtype=np.int32)
        a.AsNumpy2dInt32(result)
        self.assertTrue((A_nonzero == result).all())
        a.Zero()
        a.AsNumpy2dInt32(result)
        zeros = np.zeros((2,3), dtype=np.int32)
        self.assertTrue((zeros == result).all())

    def test_pickle(self):
        # test tensor
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            array = np.array([[[3,21,1],[22,1,5]],[[2,2,2],[7,7,7]]], dtype=dtype )
            b1 = buffers.as_tensor_buffer(array)
            pickle.dump(b1, open('tmp.pkl', 'wb'))
            b2 = pickle.load(open('tmp.pkl', 'rb'))
            array2 = buffers.as_numpy_array(b2)
            self.assertTrue((array == array2).all())

        # test matrix
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            array = np.array([[3,21,1],[22,1,5],[2,2,2],[7,7,7]], dtype=dtype )
            b1 = buffers.as_matrix_buffer(array)
            pickle.dump(b1, open('tmp.pkl', 'wb'))
            b2 = pickle.load(open('tmp.pkl', 'rb'))
            array2 = buffers.as_numpy_array(b2)
            self.assertTrue((array == array2).all())

        # test vector
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            array = np.array([3,21,1], dtype=dtype )
            b1 = buffers.as_vector_buffer(array)
            pickle.dump(b1, open('tmp.pkl', 'wb'))
            b2 = pickle.load(open('tmp.pkl', 'rb'))
            array2 = buffers.as_numpy_array(b2)
            self.assertTrue((array == array2).all())





if __name__ == '__main__':
    unittest.main()