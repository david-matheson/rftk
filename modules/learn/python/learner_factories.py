import numpy as np


import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.matrix_features as matrix_features
import rftk.try_split as try_split
import rftk.splitpoints as splitpoints
import rftk.should_split as should_split
import rftk.forest_data as forest_data
import rftk.classification as classification
import rftk.predict as predict
import learn

# prepare data -> x_float, y_class

def matrix_classification_data_prepare(**kwargs):
    bufferCollection = buffers.BufferCollection()
    if 'x' in kwargs:
        bufferCollection.AddBuffer(buffers.X_FLOAT_DATA, kwargs['x'])
    if 'classes' in kwargs:
        bufferCollection.AddBuffer(buffers.CLASS_LABELS, kwargs['classes'])
    return bufferCollection

class PredictorWrapper_32f:
    def __init__(self, forest_predictor, prepare_data):
        self.forest_predictor = forest_predictor
        self.prepare_data = prepare_data

    def predict(self, **kwargs):
        result = buffers.Float32MatrixBuffer()
        bufferCollection = self.prepare_data(**kwargs)
        self.forest_predictor.PredictYs(bufferCollection, result)
        return buffers.as_numpy_array(result)


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

def create_try_split_criteria(**kwargs):
    try_split_criteria_list = []
    min_node_size = int( kwargs.get('min_node_size', 1) )
    min_node_size_criteria = try_split.MinNodeSizeCriteria(min_node_size)
    try_split_criteria_list.append( min_node_size_criteria )
    if 'max_depth' in kwargs:
        max_depth = int( kwargs.get('max_depth') )
        max_depth_criteria = try_split.MaxDepthCriteria(max_depth)
        try_split_criteria_list.append(max_depth_criteria)
    if 'max_seconds_to_learn' in kwargs:
        max_seconds_to_learn = int( kwargs.get('max_seconds_to_learn') )
        time_limit_criteria = try_split.TimeLimitCriteria(max_seconds_to_learn)
        try_split_criteria_list.append(time_limit_criteria)
    return try_split.TrySplitCombinedCriteria(try_split_criteria_list)


def create_axis_aligned_matrix_learner_32f(**kwargs):
    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_features = int( kwargs.get('number_of_features', np.sqrt(kwargs['x'].shape[1])) )
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    number_of_classes = int( np.max(kwargs['classes']) + 1 )

    try_split_criteria = create_try_split_criteria(**kwargs)

    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_f32i32(buffers.X_FLOAT_DATA)
    else:
        sample_data_step = pipeline.AllSamplesStep_f32i32(buffers.X_FLOAT_DATA)

    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    tree_steps_pipeline = pipeline.Pipeline([sample_data_step, set_number_features_step])

    feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                      feature_params_step.IntParamsBufferId,
                                                                      sample_data_step.IndicesBufferId,
                                                                      buffers.X_FLOAT_DATA)
    matrix_feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(matrix_feature, feature_ordering)
    slice_classes_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    class_infogain_walker = classification.ClassInfoGainWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                      slice_classes_step.SlicedBufferId,
                                                                      number_of_classes)
    best_splitpint_step = classification.ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32(class_infogain_walker,
                                                                        matrix_feature_extractor_step.FeatureValuesBufferId,
                                                                        feature_ordering)
    node_steps_pipeline = pipeline.Pipeline([feature_params_step, matrix_feature_extractor_step,
                                            slice_classes_step, slice_weights_step, best_splitpint_step])

    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsBufferId,
                                                          best_splitpint_step.LeftYsBufferId,
                                                          best_splitpint_step.RightYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          matrix_feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          sample_data_step.IndicesBufferId)
    should_split_criteria = should_split.ShouldSplitNoCriteria()
    finalizer = classification.ClassEstimatorFinalizer_f32()
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer )

    tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, 5, 5, number_of_classes, number_of_jobs)
    return forest_learner


def create_matrix_predictor_32f(forest, **kwargs):
    number_of_classes = int( np.max(kwargs['classes']) + 1 )
    all_samples_step = pipeline.AllSamplesStep_f32i32(buffers.X_FLOAT_DATA)
    combiner = classification.ClassProbabilityCombiner_f32(number_of_classes)
    matrix_feature = matrix_features.LinearFloat32MatrixFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                        buffers.X_FLOAT_DATA)
    forest_predicter = predict.LinearMatrixClassificationPredictin_f32i32(forest, matrix_feature, combiner, all_samples_step)
    return PredictorWrapper_32f(forest_predicter, matrix_classification_data_prepare)


def create_vanilia_classifier():
    return LearnerWrapper(  matrix_classification_data_prepare,
                            create_axis_aligned_matrix_learner_32f,
                            create_matrix_predictor_32f)