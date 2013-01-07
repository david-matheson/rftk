import numpy as np

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
        print "sample params"
        # returns int_params and float_params

    def construct_feature_extractor(data, indices):
        print "constructing feature extractor"


