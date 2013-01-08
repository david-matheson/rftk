
class CriteriaError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class CriteriaPreSplitParams:
    def __init__(self, tree_depth, number_samples):
        self.tree_depth = tree_depth
        self.number_samples = number_samples


class CriteriaPostSplitParams:
    def __init__(self, impurity_gain, left_number_samples, right_number_samples):
        self.impurity_gain = impurity_gain
        self.left_number_samples = left_number_samples
        self.right_number_samples = right_number_samples


class DepthCriteria:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def pre_check(self, params):
        # print params.tree_depth
        if params.tree_depth > self.max_depth:
            # print "Depth"
            raise CriteriaError("Depth")

    def post_check(self, params):
        pass


class MinSamples:
    def __init__(self, min_samples):
        self.min_samples = min_samples

    def pre_check(self, params):
        # print params.number_samples
        if params.number_samples <= self.min_samples:
            # print "Min Samples"
            raise CriteriaError("Min Samples")

    def post_check(self, params):
        if params.left_number_samples <= 1:
            # print "Left Min Samples"
            raise CriteriaError("Min Samples")

        if params.right_number_samples <= 1:
            # print "Right Min Samples"
            raise CriteriaError("Min Samples")


class ImpurityGain:
    def __init__(self, min_impurity_gain):
        self.min_impurity_gain = min_impurity_gain

    def pre_check(self, params):
        pass

    def post_check(self, params):
        # print params.impurity_gain
        if params.impurity_gain <= self.min_impurity_gain:
            # print "ImpurityGain"
            raise CriteriaError("ImpurityGain") 