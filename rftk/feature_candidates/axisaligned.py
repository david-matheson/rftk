import numpy as np

import rftk.native.features as features
import rftk.native.feature_extractors as feature_extractors

class AxisAlignedFeatureCandidates:

    def __init__(self, number_of_candidates, x_dim):
        self.number_of_candidates = number_of_candidates
        self.x_dim = x_dim

    def get_int_params_dim(self):
        return 1

    def get_float_params_dim(self):
        return 1 #really should be zero but this will cause things to break

    def get_number_of_candidates(self):
        return self.number_of_candidates

    def seed(self, seed_value):
        np.random.seed(seed_value)

    def sample_params(self):
        axis = np.array( np.random.randint(self.x_dim, size=(self.number_of_candidates, self.get_int_params_dim())), dtype=np.int32)
        zeros = np.zeros((self.number_of_candidates, self.get_int_params_dim()), dtype=np.float32)
        return axis,zeros

    def construct_feature_extractor(self, data, indices):
        axis_aligned_feature_extractor = feature_extractors.AxisAlignedFeatureExtractor(data)
        return axis_aligned_feature_extractor


