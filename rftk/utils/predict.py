import numpy as np

import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.features as features
import rftk.native.predict as predict
import rftk.native.forest_data as forest_data


def as_predict_forest( tree_data_list ):
    predict_trees = [forest_data.Tree(buffers.as_matrix_buffer(t.paths),
                                    buffers.as_matrix_buffer(t.int_params),
                                    buffers.as_matrix_buffer(t.float_params),
                                    buffers.as_matrix_buffer(t.depth),
                                    buffers.as_matrix_buffer(t.counts),
                                    buffers.as_matrix_buffer(t.ys)) for t in tree_data_list]
    predict_forest = forest_data.Forest(predict_trees)
    return predict_forest

def vec_predict_ys(vec_predict_forest, x):
    (m,n) = x.shape
    buffer_collection = buffers.BufferCollection()
    buffer_collection.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(x))

    yhat_buffer = buffers.Float32MatrixBuffer()
    vec_predict_forest.PredictYs(buffer_collection, m, yhat_buffer)
    return buffers.as_numpy_array(yhat_buffer)


class MatrixForestPredictor:
    def __init__(self, forest_data):
        self.predict_forest = predict.ForestPredictor(forest_data)

    def predict_proba(self, x):
        y_probs = vec_predict_ys(self.predict_forest, x)
        return y_probs