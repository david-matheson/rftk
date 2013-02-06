
class ForestMeasurement(object):
    def __init__(self, data_config, train_config, number_of_samples, number_of_passes, accuracy):
        self.data_config = data_config
        self.train_config = train_config
        self.number_of_samples = number_of_samples
        self.number_of_passes = number_of_passes
        self.accuracy = accuracy

class OnlineForestMeasurement(ForestMeasurement):
    pass

class SklearnForestMeasurement(ForestMeasurement):
    pass

class OfflineForestMeasurement(ForestMeasurement):
    pass

class OnlineTreeMeasurement( ForestMeasurement ):
    def __init__(self, data_config, train_config, number_of_samples, number_of_passes, accuracy, tree_id):
        super(OnlineTreeMeasurement, self).__init__(data_config, train_config, number_of_samples, number_of_passes, accuracy)
        self.tree_id = tree_id

class ForestStatsMeasurement(object):
    def __init__(self, data_config, train_config, number_of_samples, number_of_passes, stats ):
        self.data_config = data_config
        self.train_config = train_config
        self.number_of_samples = number_of_samples
        self.number_of_passes = number_of_passes
        self.min_depth = stats.mMinDepth
        self.max_depth = stats.mMaxDepth
        self.average_depth = stats.GetAverageDepth()
        self.min_estimator_points = stats.mMinEstimatorPoints
        self.max_estimator_points = stats.mMaxEstimatorPoints
        self.average_estimator_points = stats.GetAverageEstimatorPoints()

class OnlineForestStatsMeasurement(ForestStatsMeasurement):
    pass

class OfflineForestStatsMeasurement(ForestStatsMeasurement):
    pass

