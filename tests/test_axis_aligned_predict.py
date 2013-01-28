import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.features as features
import rftk.native.forest_data as forest_data
import rftk.native.predict as predict
import rftk.utils.buffer_converters as buffer_converters

class TestPredictWithAxisAligned(unittest.TestCase):

    def construct_axis_aligned_forest(self):
        #                   (0) X[0] > 2.2
        #                   /           \
        #           (1) X[1] > -5     (2) [0.7 0.1 0.2]
        #           /            \
        # (3) [0.3 0.3 0.4]   (4) [0.3 0.6 0.1]
        #
        #                   (0) X[0] > 5.0
        #                   /           \
        #           (1) X[0] > 2.5     (2) [0.8 0.1 0.1]
        #           /            \
        # (3) [0.2 0.2 0.6]   (4) [0.2 0.7 0.1]
        path_1 = buffer_converters.as_matrix_buffer(np.array([[1,2],[3,4],[-1,-1],[-1,-1],[-1,-1]], dtype=np.int32))
        int_params_1 = buffer_converters.as_matrix_buffer(np.array([[1,0],[1,1],[1,0],[1,0],[1,0]], dtype=np.int32))
        float_params_1 = buffer_converters.as_matrix_buffer(np.array([[2.2],[-5],[0],[0],[0]], dtype=np.float32))
        ys_1 = buffer_converters.as_matrix_buffer(np.array([[0,0,0],[0,0,0],[0.7,0.1,0.2],[0.3,0.3,0.4],[0.3,0.6,0.1]], dtype=np.float32))
        depth_1 = buffer_converters.as_matrix_buffer(np.array([0, 1, 1, 2, 2], dtype=np.int32))
        tree_1 = forest_data.Tree(path_1, int_params_1, float_params_1, depth_1, ys_1)

        path_2 = buffer_converters.as_matrix_buffer(np.array([[1,2],[3,4],[-1,-1],[-1,-1],[-1,-1]], dtype=np.int32))
        int_params_2 = buffer_converters.as_matrix_buffer(np.array([[1,0],[1,0],[1,0],[1,0],[1,0]], dtype=np.int32))
        float_params_2 = buffer_converters.as_matrix_buffer(np.array([[5.0],[2.5],[0],[0],[0]], dtype=np.float32))
        ys_2 = buffer_converters.as_matrix_buffer(np.array([[0,0,0],[0,0,0],[0.8,0.1,0.1],[0.2,0.2,0.6],[0.2,0.7,0.1]], dtype=np.float32))
        depth_2 = buffer_converters.as_matrix_buffer(np.array([0, 1, 1, 2, 2], dtype=np.int32))
        tree_2 = forest_data.Tree(path_2, int_params_2, float_params_2, depth_2, ys_2)

        forest = forest_data.Forest([tree_1, tree_2])
        return forest


    def test_predict_leafnodes_with_axis_aligned(self):
        forest = self.construct_axis_aligned_forest()
        forest_predict = predict.ForestPredictor(forest)

        x = buffer_converters.as_matrix_buffer(np.array([[4.0,0.0]], dtype=np.float32))
        buffer_collection = buffers.BufferCollection()
        buffer_collection.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, x)
        leaf_node_ids = buffers.Int32MatrixBuffer()
        forest_predict.PredictLeafs(buffer_collection, 1, leaf_node_ids)

        self.assertEqual(leaf_node_ids.Get(0,0), 3)
        self.assertEqual(leaf_node_ids.Get(0,1), 2)

    def test_predict_ys_with_axis_aligned(self):
        forest = self.construct_axis_aligned_forest()
        forest_predict = predict.ForestPredictor(forest)

        x = buffer_converters.as_matrix_buffer(np.array([[4.0,0.0]], dtype=np.float32))
        buffer_collection = buffers.BufferCollection()
        buffer_collection.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, x)
        ys = buffers.Float32MatrixBuffer()
        forest_predict.PredictYs(buffer_collection, 1, ys)


        self.assertAlmostEqual(ys.Get(0,0), 0.55)
        self.assertAlmostEqual(ys.Get(0,1), 0.2)
        self.assertAlmostEqual(ys.Get(0,2), 0.25)


if __name__ == '__main__':
    unittest.main()