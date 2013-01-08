import rftk.native.bootstrap

class BootstapSampler:
    def __init__(self, number_of_samples, with_replacement=True):
        self.number_of_samples = number_of_samples
        self.with_replacement = with_replacement

    def sample(self, number_of_indices):
        assert(self.with_replacement or number_of_indices >= number_of_samples)
        return bootstrap.sample(number_of_indices, self.number_of_samples, self.with_replacement)

class EverythingSampler:
    def sample(self, number_of_indices):
        return np.ones(number_of_indices, dtype=np.int32)