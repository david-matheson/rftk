import numpy as np

import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.features as features
import rftk.native.predict as predict
import rftk.native.forest_data as forest_data
import rftk.utils.buffer_converters as buffer_converters

def as_predict_forest( tree_data_list ):
    predict_trees = [forest_data.Tree(buffer_converters.as_matrix_buffer(t.paths),
                                    buffer_converters.as_matrix_buffer(t.int_params),
                                    buffer_converters.as_matrix_buffer(t.float_params),
                                    buffer_converters.as_matrix_buffer(t.depth),
                                    buffer_converters.as_matrix_buffer(t.ys)) for t in tree_data_list]
    predict_forest = forest_data.Forest(predict_trees)
    return predict_forest

def vec_predict_ys(vec_predict_forest, x):
    (m,n) = x.shape
    buffer_collection = buffers.BufferCollection()
    buffer_collection.AddMatrixBufferFloat(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(x))

    yhat_buffer = buffers.MatrixBufferFloat()
    vec_predict_forest.PredictYs(buffer_collection, m, yhat_buffer)
    return buffer_converters.as_numpy_array(yhat_buffer)
