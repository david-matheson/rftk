import numpy as np

import rftk.utils.buffer_converters as buffer_converters

class HistogramLeafStatsFactory:
    def __init__(self, y_dim):
        self.y_dim = y_dim

    def get_ydim(self):
        return self.y_dim

    def construct(self, sample_weights, ys):
        return HistogramLeafStats(sample_weights, ys, self.y_dim)

class HistogramLeafStats:
    def __init__(self, sample_weights, ys, y_dim):
        self.sample_weights = sample_weights
        self.ys = buffer_converters.as_numpy_array(ys, flatten=True)
        self.y_dim = y_dim

    def get_y(self, sample_indices):
        y = np.zeros(self.y_dim, dtype=np.float32)
        for class_id in range(self.y_dim):
            sample_indices_class_id_y_mask = (self.ys[sample_indices] == class_id)
            y[class_id] = np.sum(self.sample_weights[sample_indices_class_id_y_mask])
        return (y / y.sum())


        