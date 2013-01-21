import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.utils.buffer_converters as buffer_converters
import rftk.native.best_split as best_split


class TestClassInfoGainAllThresholdsBestSplit(unittest.TestCase):

    def test_class_info_gain_all_thresholds_best_split_1_class(self):   
        data = buffers.BufferCollection()
        #feature matrix is (#features X #samples)
        data.AddMatrixBufferFloat(buffers.FEATURE_VALUES, 
            buffer_converters.as_matrix_buffer(np.array([[4,0],[3,1],[2,0],[1,1]], dtype=np.float32))) #[4,3,2,1],[0,1,0,1]]
        data.AddMatrixBufferInt(buffers.CLASS_LABELS,
            buffer_converters.as_matrix_buffer(np.array([0,0,1,1], dtype=np.int32)))
        data.AddMatrixBufferFloat(buffers.SAMPLE_WEIGHTS, 
            buffer_converters.as_matrix_buffer(np.array([1,1,1,1], dtype=np.float32)))

        splitter = best_split.ClassInfoGainAllThresholdsBestSplit(1.0, 1, data.GetMatrixBufferInt(buffers.CLASS_LABELS).GetMax()+1 )

        impurity_buffer = buffers.MatrixBufferFloat()
        threshold_buffer = buffers.MatrixBufferFloat()
        child_counts_buffer = buffers.MatrixBufferFloat()
        left_ys_buffer = buffers.MatrixBufferFloat()
        right_ys_buffer = buffers.MatrixBufferFloat()
        splitter.BestSplits(  data,
                                impurity_buffer,
                                threshold_buffer,
                                child_counts_buffer,
                                left_ys_buffer,
                                right_ys_buffer)
        impurity = buffer_converters.as_numpy_array(impurity_buffer)
        self.assertAlmostEqual(impurity[0], 0.69314718)
        self.assertAlmostEqual(impurity[1], 0.21576154)

        threshold = buffer_converters.as_numpy_array(threshold_buffer, flatten=True)
        expected_threshold = np.array([2.5,0], dtype=np.float32)
        self.assertTrue((threshold == expected_threshold).all())

        child_counts = buffer_converters.as_numpy_array(child_counts_buffer, flatten=True)
        expected_child_counts = np.array([[2,2],[3,1]], dtype=np.float32)
        self.assertTrue((child_counts == expected_child_counts).all())

        self.assertAlmostEqual(left_ys_buffer.Get(0,0), 1)
        self.assertAlmostEqual(left_ys_buffer.Get(0,1), 0)
        self.assertAlmostEqual(left_ys_buffer.Get(1,0), 1.0/3.0)
        self.assertAlmostEqual(left_ys_buffer.Get(1,1), 2.0/3.0)

        self.assertAlmostEqual(right_ys_buffer.Get(0,0), 0)
        self.assertAlmostEqual(right_ys_buffer.Get(0,1), 1)
        self.assertAlmostEqual(right_ys_buffer.Get(1,0), 1)
        self.assertAlmostEqual(right_ys_buffer.Get(1,1), 0)


    def test_class_info_gain_all_thresholds_best_split_2_class(self):
        data = buffers.BufferCollection()
        #feature matrix is (#features X #samples)
        data.AddMatrixBufferFloat(buffers.FEATURE_VALUES, 
            buffer_converters.as_matrix_buffer(np.array([[1,2,-3,4,5,6],
                                                        [0,0,0,1,1,1],
                                                        [1,10,2,11,3,12]], 
                                                        dtype=np.float32)).Transpose())
        data.AddMatrixBufferInt(buffers.CLASS_LABELS,
            buffer_converters.as_matrix_buffer(np.array([1,1,0,1,0,1], dtype=np.int32)))
        data.AddMatrixBufferFloat(buffers.SAMPLE_WEIGHTS,
            buffer_converters.as_matrix_buffer(np.array([1,1,1,1,1,1], dtype=np.float32)))
        splitter = best_split.ClassInfoGainAllThresholdsBestSplit( 1.0, 1, data.GetMatrixBufferInt(buffers.CLASS_LABELS).GetMax()+1)

        impurity_buffer = buffers.MatrixBufferFloat()
        threshold_buffer = buffers.MatrixBufferFloat()
        child_counts_buffer = buffers.MatrixBufferFloat()
        left_ys_buffer = buffers.MatrixBufferFloat()
        right_ys_buffer = buffers.MatrixBufferFloat()
        splitter.BestSplits(  data,
                                impurity_buffer,
                                threshold_buffer,
                                child_counts_buffer,
                                left_ys_buffer,
                                right_ys_buffer)
        impurity = buffer_converters.as_numpy_array(impurity_buffer)
        threshold = buffer_converters.as_numpy_array(threshold_buffer, flatten=True)
        self.assertEqual(threshold[np.argmax(impurity)], 6.5)

if __name__ == '__main__':
    unittest.main()