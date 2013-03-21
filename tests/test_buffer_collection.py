import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers


class TestBufferCollectoin(unittest.TestCase):

    def test_vector(self):
        data_1 = np.array([3,21,1,22,1,5], dtype=np.float32)
        collection = buffers.BufferCollection(first=data_1)

        self.assertEqual(collection.HasBuffer("first"), True)
        self.assertEqual(collection.HasFloat32VectorBuffer("first"), True)
        self.assertEqual(collection.HasFloat64VectorBuffer("first"), False)
        self.assertEqual(collection.HasInt32VectorBuffer("first"), False)
        self.assertEqual(collection.HasInt64VectorBuffer("first"), False)

        self.assertEqual(collection.HasBuffer("second"), False)
        data_2 = np.array([3,21,1], dtype=np.float64 )
        collection.AddBuffer("second", data_2)
        self.assertEqual(collection.HasFloat32VectorBuffer("second"), False)
        self.assertEqual(collection.HasFloat64VectorBuffer("second"), True)
        self.assertEqual(collection.HasInt32VectorBuffer("second"), False)
        self.assertEqual(collection.HasInt64VectorBuffer("second"), False)

        self.assertEqual(collection.HasBuffer("third"), False)
        data_3 = np.array([22,1,5], dtype=np.int32 )
        collection.AddBuffer("third", data_3)
        self.assertEqual(collection.HasFloat32VectorBuffer("third"), False)
        self.assertEqual(collection.HasFloat64VectorBuffer("third"), False)
        self.assertEqual(collection.HasInt32VectorBuffer("third"), True)
        self.assertEqual(collection.HasInt64VectorBuffer("third"), False)

        self.assertEqual(collection.HasBuffer("forth"), False)
        data_4 = np.array([1,1,1], dtype=np.int64 )
        collection.AddBuffer("forth", data_4)
        self.assertEqual(collection.HasFloat32VectorBuffer("forth"), False)
        self.assertEqual(collection.HasFloat64VectorBuffer("forth"), False)
        self.assertEqual(collection.HasInt32VectorBuffer("forth"), False)
        self.assertEqual(collection.HasInt64VectorBuffer("forth"), True)

        data_1_out = buffers.as_numpy_array(collection.GetBuffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffers.as_numpy_array(collection.GetBuffer("second"))
        self.assertTrue((data_2 == data_2_out).all())

        data_3_out = buffers.as_numpy_array(collection.GetBuffer("third"))
        self.assertTrue((data_3 == data_3_out).all())

        data_4_out = buffers.as_numpy_array(collection.GetBuffer("forth"))
        self.assertTrue((data_4 == data_4_out).all())


    def test_matrix(self):
        data_1 = np.array([[3,21,1],[22,1,5],[2,3,4],[7,7,7]], dtype=np.float32)
        collection = buffers.BufferCollection(first=data_1)

        self.assertEqual(collection.HasBuffer("first"), True)
        self.assertEqual(collection.HasFloat32MatrixBuffer("first"), True)
        self.assertEqual(collection.HasFloat64MatrixBuffer("first"), False)
        self.assertEqual(collection.HasInt32MatrixBuffer("first"), False)
        self.assertEqual(collection.HasInt64MatrixBuffer("first"), False)

        self.assertEqual(collection.HasBuffer("second"), False)
        data_2 = np.array([[3,21,1],[22,1,5]], dtype=np.float64 )
        collection.AddBuffer("second", data_2)
        self.assertEqual(collection.HasFloat32MatrixBuffer("second"), False)
        self.assertEqual(collection.HasFloat64MatrixBuffer("second"), True)
        self.assertEqual(collection.HasInt32MatrixBuffer("second"), False)
        self.assertEqual(collection.HasInt64MatrixBuffer("second"), False)

        self.assertEqual(collection.HasBuffer("third"), False)
        data_3 = np.array([[3,21,1],[22,1,5]], dtype=np.int32 )
        collection.AddBuffer("third", data_3)
        self.assertEqual(collection.HasFloat32MatrixBuffer("third"), False)
        self.assertEqual(collection.HasFloat64MatrixBuffer("third"), False)
        self.assertEqual(collection.HasInt32MatrixBuffer("third"), True)
        self.assertEqual(collection.HasInt64MatrixBuffer("third"), False)

        self.assertEqual(collection.HasBuffer("forth"), False)
        data_4 = np.array([[3,21,1],[1,1,1]], dtype=np.int64 )
        collection.AddBuffer("forth", data_4)
        self.assertEqual(collection.HasFloat32MatrixBuffer("forth"), False)
        self.assertEqual(collection.HasFloat64MatrixBuffer("forth"), False)
        self.assertEqual(collection.HasInt32MatrixBuffer("forth"), False)
        self.assertEqual(collection.HasInt64MatrixBuffer("forth"), True)

        data_1_out = buffers.as_numpy_array(collection.GetBuffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffers.as_numpy_array(collection.GetBuffer("second"))
        self.assertTrue((data_2 == data_2_out).all())

        data_3_out = buffers.as_numpy_array(collection.GetBuffer("third"))
        self.assertTrue((data_3 == data_3_out).all())

        data_4_out = buffers.as_numpy_array(collection.GetBuffer("forth"))
        self.assertTrue((data_4 == data_4_out).all())


    def test_tensor3(self):
        data_1 = np.array([[[3,21,1],[22,1,5]],[[2,3,4],[7,7,7]]], dtype=np.float32)
        collection = buffers.BufferCollection(first=data_1)

        self.assertEqual(collection.HasBuffer("first"), True)
        self.assertEqual(collection.HasFloat32Tensor3Buffer("first"), True)
        self.assertEqual(collection.HasFloat64Tensor3Buffer("first"), False)
        self.assertEqual(collection.HasInt32Tensor3Buffer("first"), False)
        self.assertEqual(collection.HasInt64Tensor3Buffer("first"), False)

        self.assertEqual(collection.HasBuffer("second"), False)
        data_2 = np.array([[[3,21,1],[22,1,5]]], dtype=np.float64 )
        collection.AddBuffer("second", data_2)
        self.assertEqual(collection.HasFloat32Tensor3Buffer("second"), False)
        self.assertEqual(collection.HasFloat64Tensor3Buffer("second"), True)
        self.assertEqual(collection.HasInt32Tensor3Buffer("second"), False)
        self.assertEqual(collection.HasInt64Tensor3Buffer("second"), False)

        self.assertEqual(collection.HasBuffer("third"), False)
        data_3 = np.array([[[3,21,1]],[[22,1,5]]], dtype=np.int32 )
        collection.AddBuffer("third", data_3)
        self.assertEqual(collection.HasFloat32Tensor3Buffer("third"), False)
        self.assertEqual(collection.HasFloat64Tensor3Buffer("third"), False)
        self.assertEqual(collection.HasInt32Tensor3Buffer("third"), True)
        self.assertEqual(collection.HasInt64Tensor3Buffer("third"), False)

        self.assertEqual(collection.HasBuffer("forth"), False)
        data_4 = np.array([[[3,21,1,1,1,1]]], dtype=np.int64 )
        collection.AddBuffer("forth", data_4)
        self.assertEqual(collection.HasFloat32Tensor3Buffer("forth"), False)
        self.assertEqual(collection.HasFloat64Tensor3Buffer("forth"), False)
        self.assertEqual(collection.HasInt32Tensor3Buffer("forth"), False)
        self.assertEqual(collection.HasInt64Tensor3Buffer("forth"), True)

        data_1_out = buffers.as_numpy_array(collection.GetBuffer("first"))
        self.assertTrue((data_1 == data_1_out).all())

        data_2_out = buffers.as_numpy_array(collection.GetBuffer("second"))
        self.assertTrue((data_2 == data_2_out).all())

        data_3_out = buffers.as_numpy_array(collection.GetBuffer("third"))
        self.assertTrue((data_3 == data_3_out).all())

        data_4_out = buffers.as_numpy_array(collection.GetBuffer("forth"))
        self.assertTrue((data_4 == data_4_out).all())


