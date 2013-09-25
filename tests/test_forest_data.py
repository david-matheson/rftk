import unittest as unittest
import numpy as np

import pickle

import rftk.buffers as buffers
import rftk.forest_data as forest_data

class TestForestData(unittest.TestCase):

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
        path_1 = buffers.as_matrix_buffer(np.array([[1,2],[3,4],[-1,-1],[-1,-1],[-1,-1]], dtype=np.int32))
        int_params_1 = buffers.as_matrix_buffer(np.array([[1,0],[1,1],[1,0],[1,0],[1,0]], dtype=np.int32))
        float_params_1 = buffers.as_matrix_buffer(np.array([[2.2],[-5],[0],[0],[0]], dtype=np.float32))
        ys_1 = buffers.as_matrix_buffer(np.array([[0,0,0],[0,0,0],[0.7,0.1,0.2],[0.3,0.3,0.4],[0.3,0.6,0.1]], dtype=np.float32))
        depth_1 = buffers.as_vector_buffer(np.array([0, 1, 1, 2, 2], dtype=np.int32))
        counts_1 = buffers.as_vector_buffer(np.array([5, 5, 5, 5, 5], dtype=np.float32))
        tree_1 = forest_data.Tree(path_1, int_params_1, float_params_1, depth_1, counts_1, ys_1)

        path_2 = buffers.as_matrix_buffer(np.array([[1,2],[3,4],[-1,-1],[-1,-1],[-1,-1]], dtype=np.int32))
        int_params_2 = buffers.as_matrix_buffer(np.array([[1,0],[1,0],[1,0],[1,0],[1,0]], dtype=np.int32))
        float_params_2 = buffers.as_matrix_buffer(np.array([[5.0],[2.5],[0],[0],[0]], dtype=np.float32))
        ys_2 = buffers.as_matrix_buffer(np.array([[0,0,0],[0,0,0],[0.8,0.1,0.1],[0.2,0.2,0.6],[0.2,0.7,0.1]], dtype=np.float32))
        depth_2 = buffers.as_vector_buffer(np.array([0, 1, 1, 2, 2], dtype=np.int32))
        counts_2 = buffers.as_vector_buffer(np.array([5, 5, 5, 5, 5], dtype=np.float32))
        tree_2 = forest_data.Tree(path_2, int_params_2, float_params_2, depth_2, counts_2, ys_2)
        tree_2.GetExtraInfo().AddBuffer("first", np.array([3,21,1,22,1,5], dtype=np.float32))

        forest = forest_data.Forest([tree_1, tree_2])

        return forest

    def test_tree_pickle(self):
        forest = self.construct_axis_aligned_forest()
        tree = forest.GetTree(1)
        pickle.dump(tree, open('tmp.pkl', 'wb'))
        tree2 = pickle.load(open('tmp.pkl', 'rb'))

        self.assertAlmostEqual(tree2.GetYs().Get(2,0), 0.8, places=7)
        self.assertEqual(tree2.GetCounts().Get(3), 5)

        b1 = buffers.as_numpy_array(tree.GetExtraInfo().GetBuffer("first"))
        b2 = buffers.as_numpy_array(tree2.GetExtraInfo().GetBuffer("first"))
        self.assertTrue((b1 == b2).all())

    def test_forest_pickle(self):
        forest = self.construct_axis_aligned_forest()
        pickle.dump(forest, open('tmp.pkl', 'wb'))
        forest2 = pickle.load(open('tmp.pkl', 'rb'))

        tree = forest.GetTree(1)
        tree2 = forest2.GetTree(1)
        self.assertAlmostEqual(tree2.GetYs().Get(2,0), 0.8, places=7)
        self.assertEqual(tree2.GetCounts().Get(3), 5)

        b1 = buffers.as_numpy_array(tree.GetExtraInfo().GetBuffer("first"))
        b2 = buffers.as_numpy_array(tree2.GetExtraInfo().GetBuffer("first"))
        self.assertTrue((b1 == b2).all())
