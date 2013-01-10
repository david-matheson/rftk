import numpy as np

import rftk.native.assert_util
import rftk.native.buffers as buffers
import rftk.native.features as features
import rftk.native.predict as predict
import rftk.utils.buffer_converters as buffer_converters

def as_predict_forest( tree_data_list ):
    predict_trees = [predict.Tree(buffer_converters.as_matrix_buffer(t.paths), 
                                    buffer_converters.as_matrix_buffer(t.int_params),
                                    buffer_converters.as_matrix_buffer(t.float_params),
                                    buffer_converters.as_matrix_buffer(t.ys)) for t in tree_data_list]
    predict_forest = predict.Forest(predict_trees)
    return predict_forest

def vec_predict_ys(vec_predict_forest, x):
    x_buffer = buffer_converters.as_matrix_buffer(x)
    yhat_buffer = buffers.MatrixBufferFloat()
    vec_predict_forest.PredictYs(x_buffer, yhat_buffer)
    return buffer_converters.as_numpy_array(yhat_buffer)
