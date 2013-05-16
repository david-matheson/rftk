import numpy as np

import rftk.buffers as buffers
import rftk.features as features
import rftk.forest_data as forest_data

import predict


class MatrixForestPredictor:
    def __init__(self, forest_data):
        self.forest_data = forest_data

    def predict_proba(self, x):
        buffer_collection = buffers.BufferCollection()
        buffer_collection.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(x))

        number_of_classes = self.forest_data.GetTree(0).mYs.GetN()
        all_samples_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
        combiner = classification.ClassProbabilityCombiner_f32(number_of_classes)
        matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                            buffers.X_FLOAT_DATA)
        forest_predicter = predict.LinearMatrixClassificationPredictin_f32i32(forest_data, matrix_feature, combiner, all_samples_step)

        result = buffers.Float32MatrixBuffer()
        forest_predicter.PredictYs(bufferCollection, result)
        return buffers.as_numpy_array(result)
