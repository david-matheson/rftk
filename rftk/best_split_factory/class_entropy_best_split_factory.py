import rftk.native.best_split as best_split
import rftk.utils.buffer_converters as buffer_converters

class ClassEntropyBestSplitFactory:
    def __init__(self, ratio_of_thresholds_to_test, min_thresholds_to_test, max_class):
        self.ratio_of_thresholds_to_test = ratio_of_thresholds_to_test
        self.min_thresholds_to_test = min_thresholds_to_test
        self.max_class = max_class

    def construct(self):
        return best_split.ClassInfoGainAllThresholdsBestSplit(  self.ratio_of_thresholds_to_test,
                                                                self.min_thresholds_to_test,
                                                                self.max_class)