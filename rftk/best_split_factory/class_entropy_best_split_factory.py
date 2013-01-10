import rftk.native.best_split as best_split
import rftk.utils.buffer_converters as buffer_converters

class ClassEntropyBestSplitFactory:
    def __init__(self, ratio_of_thresholds_to_test, min_thresholds_to_test):
        self.ratio_of_thresholds_to_test = ratio_of_thresholds_to_test
        self.min_thresholds_to_test = min_thresholds_to_test

    def construct(self, sample_weights, ys):
        sample_weights_buffer = buffer_converters.as_matrix_buffer(sample_weights)
        return best_split.ClassInfoGainAllThresholdsBestSplit(  ys, 
                                                                sample_weights_buffer, 
                                                                self.ratio_of_thresholds_to_test,
                                                                self.min_thresholds_to_test)