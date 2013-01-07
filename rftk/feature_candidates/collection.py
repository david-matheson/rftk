
class FeatureParamsRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class FeatureCandidateCollection:
    def __init__(self, list_of_feature_candidates):
        self.list_of_feature_candidates = list_of_feature_candidates
        print "init"

    def seed(self, seed_value):
        for feature_candidate in self.list_of_feature_candidates:
            feature_candidate.seed( seed_value )

    def max_int_params_dim(self):
        max_dim = max(fc.get_int_params_dim() for fc in self.list_of_feature_candidates)
        return max_dim

    def max_float_params_dim(self):
        max_dim = max(fc.get_int_params_dim() for fc in self.list_of_feature_candidates)
        return max_dim

    def number_of_candidates(self):
        number_of_candidates = sum(fc.get_number_of_candidates() for fc in self.list_of_feature_candidates)

    def sample_params(self):
        number_of_candidates = self.number_of_candidates()
        int_params_dim = self.max_int_params_dim()
        float_params_dim = self.max_float_params_dim()

        int_params = np.zeros((number_of_candidates, int_params_dim), dtype=np.int32)
        float_params = np.zeros((number_of_candidates, float_params_dim), dtype=np.float32)
        feature_params_range_list = []

        start = 0

        for fc in self.list_of_feature_candidates:
            end = start + fc.get_number_of_candidates()
            fc_int_params, fc_float_params = fc.sample_params()
            int_params[start:end,1:fc.get_int_params_dim()+1] = fc_int_params[:,0:fc.get_int_params_dim()]
            float_params[start:end,1:fc.get_float_params_dim()+1] = fc_float_params[:,0:fc.get_float_params_dim()]
            feature_params_range_list.append(FeatureParamsRange(start=start, end=end))

            # set start for next feature candidates
            start = end

        return int_params, float_params, feature_params_range_list

    def construct_feature_extractor_list(data, indices):
        return [fc.construct_feature_extractor(data, indices) for fc in self.list_of_feature_candidates]