import unittest as unittest
import numpy as np
import rftk.native.bootstrap
import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.utils.buffer_converters as buffer_converters
import rftk.native.best_split as best_split


class TestClassInfoGainAllThresholdsBestSplit(unittest.TestCase):

    def test_class_info_gain_all_thresholds_best_split_2_class(self):
        #feature matrix is (#features X #samples)
        features = buffer_converters.as_matrix_buffer(np.array([[4,3,2,1],[0,1,0,1]], dtype=np.float32))

        data = buffers.BufferCollection()
        data.AddMatrixBufferInt("ClassLabels",  buffer_converters.as_matrix_buffer(np.array([0,0,1,1], dtype=np.int32)))
        data.AddMatrixBufferFloat("SampleWeights", buffer_converters.as_matrix_buffer(np.array([1,1,1,1], dtype=np.float32)))

        splitter = best_split.ClassInfoGainAllThresholdsBestSplit(1.0, 1, data.AddMatrixBufferInt("ClassLabels").GetMax() )

        sample_indices = buffer_converters.as_matrix_buffer(np.array(range(4), dtype=np.int32))
        impurity_buffer = buffers.MatrixBufferFloat()
        threshold_buffer = buffers.MatrixBufferFloat()
        splitter.BestSplits(  data,
                                sample_indices,
                                features,
                                impurity_buffer,
                                threshold_buffer)
        impurity = buffer_converters.as_numpy_array(impurity_buffer)
        self.assertAlmostEqual(impurity[0], 0.69314718)
        self.assertAlmostEqual(impurity[1], 0.21576154)

        threshold = buffer_converters.as_numpy_array(threshold_buffer, flatten=True)
        expected_threshold = np.array([2.5,0], dtype=np.float32)
        self.assertTrue((threshold == expected_threshold).all())


    def test_class_info_gain_all_thresholds_best_split_2_class(self):
        #feature matrix is (#features X #samples)
        features = buffer_converters.as_matrix_buffer(np.array([[1,2,-3,4,5,6],[0,0,0,1,1,1],[1,10,2,11,3,12]], dtype=np.float32))

        data = buffers.BufferCollection()
        data.AddMatrixBufferInt("ClassLabels",  buffer_converters.as_matrix_buffer(np.array([1,1,0,1,0,1], dtype=np.int32)))
        data.AddMatrixBufferFloat("SampleWeights", buffer_converters.as_matrix_buffer(np.array([1,1,1,1,1,1], dtype=np.float32)))
        splitter = best_split.ClassInfoGainAllThresholdsBestSplit( 1.0, 1, data.GetMatrixBufferInt("ClassLabels").GetMax())

        sample_indices = buffer_converters.as_matrix_buffer(np.array(range(6), dtype=np.int32))
        impurity_buffer = buffers.MatrixBufferFloat()
        threshold_buffer = buffers.MatrixBufferFloat()
        splitter.BestSplits(  data,
                                sample_indices,
                                features,
                                impurity_buffer,
                                threshold_buffer)
        impurity = buffer_converters.as_numpy_array(impurity_buffer)
        threshold = buffer_converters.as_numpy_array(threshold_buffer, flatten=True)
        self.assertEqual(threshold[np.argmax(impurity)], 6.5)

if __name__ == '__main__':
    unittest.main()