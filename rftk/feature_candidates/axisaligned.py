import numpy as np

import rftk.native.features as features
import rftk.native.feature_extractors as feature_extractors
import rftk.utils.buffer_converters as buffer_converters

class AxisAlignedFeatureCandidates:

    def __init__(self, number_of_candidates, x_dim):
        self.number_of_candidates = number_of_candidates
        self.x_dim = x_dim

    def get_int_params_dim(self):
        axis_aligned_feature_extractor = feature_extractors.AxisAlignedFeatureExtractor(self.number_of_candidates, self.x_dim)
        return axis_aligned_feature_extractor.GetIntParamsDim()

    def get_float_params_dim(self):
        axis_aligned_feature_extractor = feature_extractors.AxisAlignedFeatureExtractor(self.number_of_candidates, self.x_dim)
        return axis_aligned_feature_extractor.GetFloatParamsDim()

    def get_number_of_candidates(self):
        return self.number_of_candidates

    def seed(self, seed_value):
        np.random.seed(seed_value)

    def sample_params(self):
        axis_aligned_feature_extractor = feature_extractors.AxisAlignedFeatureExtractor(self.number_of_candidates, self.x_dim)
        ints = axis_aligned_feature_extractor.CreateIntParams()
        floats = axis_aligned_feature_extractor.CreateFloatParams()
        return buffer_converters.as_numpy_array(ints), buffer_converters.as_numpy_array(floats)

    def construct_feature_extractor(self, data, indices):
        axis_aligned_feature_extractor = feature_extractors.AxisAlignedFeatureExtractor(self.number_of_candidates, self.x_dim)
        return axis_aligned_feature_extractor


