import rftk.buffers as buffers

class LearnerWrapper:
    def __init__(self, prepare_data, create_learner, create_predictor ):
        self.prepare_data = prepare_data
        self.create_learner = create_learner
        self.create_predictor = create_predictor

    def fit(self, **kwargs):
        learner = self.create_learner(**kwargs)
        bufferCollection = self.prepare_data(**kwargs)
        forest = learner.Learn(bufferCollection)
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