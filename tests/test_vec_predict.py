import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.features as features
import rftk.native.predict as predict
import rftk.utils.buffer_converters as buffer_converters

class TestVecPredictWithAxisAligned(unittest.TestCase):

    def test_vec_predict_with_axis_aligned(self):
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
        path_1 = buffer_converters.as_matrix_buffer(np.array([[]], dtype=np.int32))

        data = buffer_converters.as_matrix_buffer(np.array([[2.0,21,1],[3.0,22,5]], dtype=np.float32))
        axis_aligned_feature_extractor = feature_extractors.AxisAlignedFeatureExtractor(data)

        sample_indices = buffer_converters.as_matrix_buffer(np.array(range(2), dtype=np.int32))
        int_feature_params = buffer_converters.as_matrix_buffer(np.array([1,0,2,0], dtype=np.int32))
        float_feature_params = buffer_converters.as_matrix_buffer(np.array([0,0,0,0], dtype=np.float32))
        results_buffer = buffers.MatrixBufferFloat()
        axis_aligned_feature_extractor.Extract(sample_indices,
                                                int_feature_params,
                                                float_feature_params,
                                                results_buffer)
        results = buffer_converters.as_numpy_array(results_buffer)
        expected_results = np.array([[21,22],[2,3],[1,5],[2,3]], dtype=np.float32)

        self.assertTrue((results == expected_results).all())



if __name__ == '__main__':
    unittest.main()