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

        self.assertEqual(collection.HasMatrixBufferFloat("first"), False)
        data_1 = np.array([[3,21,1],[22,1,5],[2,3,4],[7,7,7]], dtype=np.float32 )
        collection.AddMatrixBufferFloat("first", buffers.matrixBufferFloat( data_1 ))
        self.assertEqual(collection.HasMatrixBufferFloat("first"), True)

        self.assertEqual(collection.HasMatrixBufferFloat("second"), False)
        data_2 = np.array([[3,21,1],[22,1,5]], dtype=np.float32 )
        collection.AddMatrixBufferFloat("second", buffers.matrixBufferFloat( data_2 ))
        self.assertEqual(collection.HasMatrixBufferFloat("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetMatrixBufferFloat("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetMatrixBufferFloat("second"))
        self.assertTrue((data_2 == data_2_out).all())

    def test_matrix_int(self):
        collection = buffers.BufferCollection()

        self.assertEqual(collection.HasMatrixBufferInt("first"), False)
        data_1 = np.array([[3,21,1],[22,1,5],[2,3,4],[7,7,7]], dtype=np.int32 )
        collection.AddMatrixBufferInt("first", buffers.matrixBufferInt( data_1 ))
        self.assertEqual(collection.HasMatrixBufferInt("first"), True)

        self.assertEqual(collection.HasMatrixBufferInt("second"), False)
        data_2 = np.array([[3,21,1],[22,1,5]], dtype=np.int32 )
        collection.AddMatrixBufferInt("second", buffers.matrixBufferInt( data_2 ))
        self.assertEqual(collection.HasMatrixBufferInt("second"), True)

        data_1_out = buffer_converters.as_numpy_array(buffer=collection.GetMatrixBufferInt("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffer_converters.as_numpy_array(buffer=collection.GetMatrixBufferInt("second"))
        self.assertTrue((data_2 == data_2_out).all())
