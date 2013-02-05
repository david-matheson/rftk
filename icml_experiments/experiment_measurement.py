
class ForestMeasurement(object):
    def __init__(self, data_config, train_config,
                        number_of_samples, number_of_passes, accuracy):
        self.data_config = data_config
        self.train_config = train_config
        self.number_of_samples = number_of_samples
        self.number_of_passes = number_of_passes
        self.accuracy = accuracy

class OnlineForestMeasurement(ForestMeasurement):
    pass

class SklearnForestMeasurement(ForestMeasurement):
    pass

class OnlineTreeMeasurement( ForestMeasurement ):
    def __init__(self, data_config, train_config, number_of_samples, number_of_passes, accuracy, tree_id):
        super(OnlineTreeMeasurement, self).__init__(data_config, train_config, number_of_samples, number_of_passes, accuracy)
        self.tree_id = tree_id

