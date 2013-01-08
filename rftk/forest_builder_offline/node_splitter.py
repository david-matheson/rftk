import numpy as np

import rftk.native.buffers as buffers
import rftk.stop_criteria.criteria as criteria
import rftk.utils.buffer_converters as buffer_converters

class NodeSplitterInitParams:
    def __init__(   self, 
                    feature_candidate_collection, 
                    best_splitter_factory,
                    stop_criteria_list ):
        self.feature_candidate_collection = feature_candidate_collection
        self.best_splitter_factory = best_splitter_factory
        self.stop_criteria_list = stop_criteria_list

    def set_seed(self, value):
        self.feature_candidate_collection.set_seed(value)

class NodeSplitter:
    def __init__(  self, init_params, data, indices, sample_weights, ys):
        self.feature_extractors = init_params.feature_candidate_collection.construct_feature_extractor_list(data, indices)
        self.feature_candidate_collection = init_params.feature_candidate_collection
        self.best_splitter = init_params.best_splitter_factory.construct(sample_weights, ys)
        self.stop_criteria_list = init_params.stop_criteria_list

    def split(self, sample_indices, tree_depth):
        # Do pre split checks
        pre_split_params = criteria.CriteriaPreSplitParams(tree_depth=tree_depth, number_samples=len(sample_indices))
        for stop_criteria in self.stop_criteria_list:
            stop_criteria.pre_check(pre_split_params)

        # Compute all impurity values for all features
        number_of_candidates = self.feature_candidate_collection.number_of_candidates()
        int_params_dim = self.feature_candidate_collection.max_int_params_dim()
        float_params_dim = self.feature_candidate_collection.max_float_params_dim()

        int_params, float_params, feature_ranges = self.feature_candidate_collection.sample_params()
        (int_params_m, int_params_n) = int_params.shape
        (float_params_m, float_params_n) = float_params.shape
        assert(number_of_candidates == int_params_m)
        assert(number_of_candidates == float_params_m)
        assert(int_params_dim == int_params_n)
        assert(float_params_dim == float_params_n)
        impurity = np.zeros(number_of_candidates, dtype=np.float32)
        threshold = np.zeros(number_of_candidates, dtype=np.float32)
        assert(len(self.feature_extractors) == len(feature_ranges))
        sample_indices_buffer = buffer_converters.as_matrix_buffer(sample_indices)
        per_param_feature_extractors = []
        for i, extractor in enumerate(self.feature_extractors):
            r = feature_ranges[i]
            per_param_feature_extractors.extend([extractor for i in range(r.start, r.end)])

            feature_int_params_buffer = buffer_converters.as_matrix_buffer(int_params[r.start:r.end, :])
            feature_float_params_buffer = buffer_converters.as_matrix_buffer(float_params[r.start:r.end, :])
            feature_values_buffer = buffers.MatrixBufferFloat()
            extractor.Extract(sample_indices_buffer,
                            feature_int_params_buffer,
                            feature_float_params_buffer,
                            feature_values_buffer)

            impurity_buffer = buffers.MatrixBufferFloat()
            threshold_buffer = buffers.MatrixBufferFloat()
            self.best_splitter.BestSplits(sample_indices_buffer,
                                          feature_values_buffer,
                                          impurity_buffer,
                                          threshold_buffer)
            impurity[r.start:r.end] = buffer_converters.as_numpy_array(impurity_buffer)
            threshold[r.start:r.end] = buffer_converters.as_numpy_array(threshold_buffer)
        
        assert(number_of_candidates == len(impurity))
        assert(number_of_candidates == len(threshold))
        assert(number_of_candidates == len(per_param_feature_extractors))

        # Find best
        best_impurity_index = np.argmax(impurity)
        best_impurity_value = impurity[best_impurity_index]
        best_threshold_value = threshold[best_impurity_index]
        best_extractor = per_param_feature_extractors[best_impurity_index]
        
        best_int_params_with_feature_id = np.zeros((1,int_params_dim+1), dtype=np.int32)
        best_int_params_with_feature_id[0,0] = best_extractor.GetUID()
        best_int_params_with_feature_id[0,1:int_params_dim+1] = int_params[best_impurity_index, :]
        best_float_params_with_threshold = np.zeros((1,float_params_dim+1), dtype=np.float32)
        best_float_params_with_threshold[0,0] = best_threshold_value
        best_float_params_with_threshold[0,1:float_params_dim+1] = float_params[best_impurity_index, :]

        # Extract feature values to determine left and right sets
        feature_int_params_buffer = buffer_converters.as_matrix_buffer(int_params[best_impurity_index, :].reshape(1,int_params_dim))
        feature_float_params_buffer = buffer_converters.as_matrix_buffer(float_params[best_impurity_index, :].reshape(1,float_params_dim))
        feature_values_buffer = buffers.MatrixBufferFloat()
        extractor.Extract(sample_indices_buffer,
                        feature_int_params_buffer,
                        feature_float_params_buffer,
                        feature_values_buffer)
        feature_values = buffer_converters.as_numpy_array(feature_values_buffer).flatten()
        sample_indices_left = sample_indices[ feature_values > best_threshold_value ]
        sample_indices_right = sample_indices[ feature_values <= best_threshold_value ]

        # Do post split checks
        post_split_params = criteria.CriteriaPostSplitParams(   impurity_gain=best_impurity_value, 
                                                                left_number_samples=len(sample_indices_left),
                                                                right_number_samples=len(sample_indices_right))
        for stop_criteria in self.stop_criteria_list:
            stop_criteria.post_check(post_split_params)

        return best_int_params_with_feature_id, best_float_params_with_threshold, sample_indices_left, sample_indices_right
