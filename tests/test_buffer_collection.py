import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.utils.buffer_converters as buffer_converters


class TestBufferCollectoin(unittest.TestCase):

    def test_img_float(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasFloat32Tensor3Buffer("first"), False)
        data_1 = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.float32 )
        collection.AddFloat32Tensor3Buffer("first", buffers.Float32Tensor3( data_1 ))
        self.assertEqual(collection.HasFloat32Tensor3Buffer("first"), True)

        self.assertEqual(collection.HasFloat32Tensor3Buffer("second"), False)
        data_2 = np.array([[[3,21,1],[22,1,5]]], dtype=np.float32 )
        collection.AddFloat32Tensor3Buffer("second", buffers.Float32Tensor3( data_2 ))
        self.assertEqual(collection.HasFloat32Tensor3Buffer("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetFloat32Tensor3Buffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetFloat32Tensor3Buffer("second"))
        self.assertTrue((data_2 == data_2_out).all())

    def test_img_int(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasInt32Tensor3Buffer("first"), False)
        data_1 = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.int32 )
        collection.AddInt32Tensor3Buffer("first", buffers.Int32Tensor3( data_1 ))
        self.assertEqual(collection.HasInt32Tensor3Buffer("first"), True)

        self.assertEqual(collection.HasInt32Tensor3Buffer("second"), False)
        data_2 = np.array([[[3,21,1],[22,1,5]]], dtype=np.int32 )
        collection.AddInt32Tensor3Buffer("second", buffers.Int32Tensor3( data_2 ))
        self.assertEqual(collection.HasInt32Tensor3Buffer("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetInt32Tensor3Buffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetInt32Tensor3Buffer("second"))
        self.assertTrue((data_2 == data_2_out).all())

    def test_matrix_float(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasFloat32MatrixBuffer("first"), False)
        data_1 = np.array([[3,21,1],[22,1,5],[2,3,4],[7,7,7]], dtype=np.float32 )
        collection.AddFloat32MatrixBuffer("first", buffers.Float32Matrix( data_1 ))
        self.assertEqual(collection.HasFloat32MatrixBuffer("first"), True)

        self.assertEqual(collection.HasFloat32MatrixBuffer("second"), False)
        data_2 = np.array([[3,21,1],[22,1,5]], dtype=np.float32 )
        collection.AddFloat32MatrixBuffer("second", buffers.Float32Matrix( data_2 ))
        self.assertEqual(collection.HasFloat32MatrixBuffer("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetFloat32MatrixBuffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetFloat32MatrixBuffer("second"))
        self.assertTrue((data_2 == data_2_out).all())

    def test_matrix_int(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasInt32MatrixBuffer("first"), False)
        data_1 = np.array([[3,21,1],[22,1,5],[2,3,4],[7,7,7]], dtype=np.int32 )
        collection.AddInt32MatrixBuffer("first", buffers.Int32Matrix( data_1 ))
        self.assertEqual(collection.HasInt32MatrixBuffer("first"), True)

        self.assertEqual(collection.HasInt32MatrixBuffer("second"), False)
        data_2 = np.array([[3,21,1],[22,1,5]], dtype=np.int32 )
        collection.AddInt32MatrixBuffer("second", buffers.Int32Matrix( data_2 ))
        self.assertEqual(collection.HasInt32MatrixBuffer("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetInt32MatrixBuffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetInt32MatrixBuffer("second"))
        self.assertTrue((data_2 == data_2_out).all())
