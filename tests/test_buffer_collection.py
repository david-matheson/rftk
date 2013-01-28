import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.utils.buffer_converters as buffer_converters


class TestBufferCollectoin(unittest.TestCase):

    def test_img_float(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasImgBufferFloat("first"), False)
        data_1 = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.float32 )
        collection.AddImgBufferFloat("first", buffers.imgsBufferFloat( data_1 ))
        self.assertEqual(collection.HasImgBufferFloat("first"), True)

        self.assertEqual(collection.HasImgBufferFloat("second"), False)
        data_2 = np.array([[[3,21,1],[22,1,5]]], dtype=np.float32 )
        collection.AddImgBufferFloat("second", buffers.imgsBufferFloat( data_2 ))
        self.assertEqual(collection.HasImgBufferFloat("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetImgBufferFloat("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetImgBufferFloat("second"))
        self.assertTrue((data_2 == data_2_out).all())

    def test_img_int(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasImgBufferInt("first"), False)
        data_1 = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.int32 )
        collection.AddImgBufferInt("first", buffers.imgsBufferInt( data_1 ))
        self.assertEqual(collection.HasImgBufferInt("first"), True)

        self.assertEqual(collection.HasImgBufferInt("second"), False)
        data_2 = np.array([[[3,21,1],[22,1,5]]], dtype=np.int32 )
        collection.AddImgBufferInt("second", buffers.imgsBufferInt( data_2 ))
        self.assertEqual(collection.HasImgBufferInt("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetImgBufferInt("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetImgBufferInt("second"))
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
