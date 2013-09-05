import rftk.buffers as buffers

class LearnerWrapper:
    def __init__(self, prepare_data, create_learner, create_predictor, kwargs ):
        self.prepare_data = prepare_data
        self.create_learner = create_learner
        self.create_predictor = create_predictor
        self.learner = None
        self.init_kwargs = kwargs

    def fit(self, **kwargs):
        if self.learner is None:
            all_kwargs = dict(self.init_kwargs.items() + kwargs.items())
            self.learner = self.create_learner(**all_kwargs)
        bufferCollection = self.prepare_data(**kwargs)
        forest = self.learner.Learn(bufferCollection)
        forest_predictor_wrapper = self.create_predictor(forest, **kwargs)
        return forest_predictor_wrapper


class PredictorWrapper_32f:
    def __init__(self, forest_predictor, prepare_data):
        self.forest_predictor = forest_predictor
        self.prepare_data = prepare_data

    def predict(self, **kwargs):
        result = buffers.Float32MatrixBuffer()
        bufferCollection = self.prepare_data(**kwargs)
        self.forest_predictor.PredictYs(bufferCollection, result)
        return buffers.as_numpy_array(result)

    def predict_oob(self, **kwargs):
        result = buffers.Float32MatrixBuffer()
        bufferCollection = self.prepare_data(**kwargs)
        self.forest_predictor.PredictOobYs(bufferCollection, result)
        return buffers.as_numpy_array(result)

    def get_forest(self):
        return self.forest_predictor.GetForest()
