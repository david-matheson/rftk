import unittest as unittest
import numpy as np
import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.bootstrap as bootstrap
import rftk.native.features
import rftk.native.feature_extractors as feature_extractors
import rftk.utils.buffer_converters as buffer_converters

class TestDepthScaledDepthDeltaFeature(unittest.TestCase):

    def test_depth_scaled_depth_delta_feature_extractor(self):
        depth_delta_feature_extractor = feature_extractors.DepthScaledDepthDeltaFeatureExtractor(50.0, 100.0, 10, True )
        # depth_delta_feature_extractor.CreateFloatParams(5).Print()
        # depth_delta_feature_extractor.CreateIntParams(5).Print()        

        bufferCollection = buffers.BufferCollection()
        depths = buffer_converters.as_tensor_buffer(np.array([[[2.0, 2.0, 3.0, 1.0],
                                                              [2.0, 1.0, 1.0, 5.0],
                                                              [2.0, 1.0, 1.0, 1.0]]], dtype=np.float32))
        bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, depths)
        offset_scales = buffer_converters.as_matrix_buffer(np.array([ [1.0, 1.0],
                                                                      [1.0, 1.0],
                                                                      [1.0, 1.0],
                                                                      [1.0, 1.0]], dtype=np.float32))
        bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, offset_scales)  
        pixel_indices = buffer_converters.as_matrix_buffer(np.array([ [0, 0, 1],
                                                                      [0, 2, 0],
                                                                      [0, 0, 1],
                                                                      [0, 0, 1]], dtype=np.int32))
        bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, pixel_indices)  

        int_feature_params = buffer_converters.as_matrix_buffer(np.array([[1],[1]], dtype=np.int32))
        float_feature_params = buffer_converters.as_matrix_buffer(np.array([[-1.0, 0.0, 1.0, 0.0],
                                                                            [1.0, 2.0, -1.0, -2.0]], dtype=np.float32))

        sample_indices = buffers.Int32Vector(np.array(range(4), dtype=np.int32))

        results_buffer = buffers.Float32MatrixBuffer()
        depth_delta_feature_extractor.Extract(bufferCollection,
                                                sample_indices,
                                                int_feature_params,
                                                float_feature_params,
                                                results_buffer)
        results = buffer_converters.as_numpy_array(results_buffer.Transpose())
        expected_results = np.array([[1.0, 0.0, 1.0, 1.0], [3.0, -1.0, 3.0, 3.0]], dtype=np.float32)

        self.assertTrue((results == expected_results).all())



if __name__ == '__main__':
    unittest.main()