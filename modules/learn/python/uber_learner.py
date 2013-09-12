import numpy as np

import rftk.buffers as buffers
import rftk.pipeline as pipeline
import rftk.matrix_features as matrix_features
import rftk.splitpoints as splitpoints
import rftk.classification as classification
import rftk.predict as predict
import learn
from wrappers import *
from greedy_add_swap_learner import *
from split_criteria import *

def make_uber_data_prepare(learner_kwargs):

    def uber_data_prepare(**kwargs):
        bufferCollection = buffers.BufferCollection()

        # add input data buffers
        data_type = learner_kwargs.get('data_type')
        if data_type == 'matrix':
            bufferCollection.AddBuffer(buffers.X_FLOAT_DATA, kwargs['x'])
        elif data_type == 'depth_image':
            bufferCollection.AddBuffer(buffers.DEPTH_IMAGES, kwargs['depth_images'])
            bufferCollection.AddBuffer(buffers.PIXEL_INDICES, kwargs['pixel_indices'])
            if 'offset_scales' in kwargs:
                bufferCollection.AddBuffer(buffers.OFFSET_SCALES, kwargs['offset_scales'])
        else:
            raise Exception("unknown data_type")

        # add y data buffers
        prediction_type = learner_kwargs.get('prediction_type')
        if prediction_type == 'classification':
            if 'classes' in kwargs:
                bufferCollection.AddBuffer(buffers.CLASS_LABELS, kwargs['classes'])
        elif prediction_type == 'regression':
            if 'y' in kwargs:
                bufferCollection.AddBuffer(buffers.YS, kwargs['y'])
        else:
            raise Exception("unknown prediction_type")

        return bufferCollection

    return uber_data_prepare 

def make_uber_create_predictor(learner_kwargs):

    def uber_create_predictor(forest, **kwargs):

        data_type = learner_kwargs.get('data_type')
        if data_type == 'matrix':
            all_samples_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
            feature = matrix_features.LinearFloat32MatrixFeature_f32i32(all_samples_step.IndicesBufferId, buffers.X_FLOAT_DATA)
        elif data_type == 'depth_image':
            all_samples_step = pipeline.AllSamplesStep_i32f32i32(buffers.PIXEL_INDICES)
            feature = image_features.ScaledDepthDeltaFeature_f32i32(all_samples_step.IndicesBufferId,
                                                                    buffers.PIXEL_INDICES,
                                                                    buffers.DEPTH_IMAGES)
        else:
            raise Exception("unknown data_type")

        prediction_type = learner_kwargs.get('prediction_type')
        if prediction_type == 'classification':
            number_of_classes = forest.GetTree(0).mYs.GetN()    
            combiner = classification.ClassProbabilityCombiner_f32(number_of_classes)   
        elif prediction_type == 'regression':
            dimension_of_y = forest.GetTree(0).mYs.GetN() / 2
            combiner = regression.MeanVarianceCombiner_f32(dimension_of_y)
        else:
            raise Exception("unknown prediction_type")

        if data_type == 'matrix' and prediction_type == 'classification':
            forest_predicter = predict.LinearMatrixClassificationPredictin_f32i32(forest, feature, combiner, all_samples_step)
        elif data_type == 'matrix' and prediction_type == 'regression':
            forest_predicter = predict.LinearMatrixRegressionPredictin_f32i32(forest, feature, combiner, all_samples_step)
        elif data_type == 'depth_image' and prediction_type == 'classification':
            forest_predicter = predict.ScaledDepthDeltaClassificationPredictin_f32i32(forest, feature, combiner, all_samples_step)
        elif data_type == 'depth_image' and prediction_type == 'regression':
            forest_predicter = predict.ScaledDepthDeltaRegressionPrediction_f32i32(forest, feature, combiner, all_samples_step)

        return PredictorWrapper_32f(forest_predicter, make_uber_data_prepare(learner_kwargs))

    return uber_create_predictor


def uber_create_learner(**kwargs):
    # forest_steps = []
    tree_steps = []
    node_steps = []

    data_type = kwargs.get('data_type')
    prediction_type = kwargs.get('prediction_type')
    split_type = kwargs.get('split_type')
    tree_type = kwargs.get('tree_type')
    feature_ordering = int( kwargs.get('feature_ordering', pipeline.FEATURES_BY_DATAPOINTS) )
    
    # Setup sampling of weights step
    if 'bootstrap' in kwargs and kwargs.get('bootstrap'):
        sample_data_step = pipeline.BootstrapSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
    elif 'poisson_sample' in kwargs:
        poisson_sample_mean = float(kwargs.get('poisson_sample'))
        sample_data_step = pipeline.PoissonSamplesStep_f32i32(buffers.X_FLOAT_DATA, poisson_sample_mean)
    else:
        sample_data_step = pipeline.AllSamplesStep_f32f32i32(buffers.X_FLOAT_DATA)
    tree_steps.append(sample_data_step)

    # Default number of features to extract
    if data_type == 'matrix' and prediction_type == 'classification':
        default_number_of_features = int(np.sqrt(kwargs['x'].shape[1]))
    elif data_type == 'matrix' and prediction_type == 'regression':
        default_number_of_features = int(kwargs['x'].shape[1]/3 + 0.5)
    elif data_type == 'depth_image':
        default_number_of_features = 1
    else:
        raise Exception("unknown data_type")

    # Set number of features to extract
    # need to add possion and uniform
    number_of_features = int( kwargs.get('number_of_features', default_number_of_features) )
    number_of_features_buffer = buffers.as_vector_buffer(np.array([number_of_features], dtype=np.int32))
    set_number_features_step = pipeline.SetInt32VectorBufferStep(number_of_features_buffer, pipeline.WHEN_NEW)
    node_steps.append(set_number_features_step)

    # Setup extraction step
    if data_type == 'matrix':
        extractor_type = kwargs.get('extractor_type')
        if extractor_type == 'axis_aligned':
            feature_params_step = matrix_features.AxisAlignedParamsStep_f32i32(set_number_features_step.OutputBufferId, buffers.X_FLOAT_DATA)
        else:
            raise Exception("unknown extractor_type")

        feature = matrix_features.LinearFloat32MatrixFeature_f32i32(feature_params_step.FloatParamsBufferId,
                                                                  feature_params_step.IntParamsBufferId,
                                                                  sample_data_step.IndicesBufferId,
                                                                  buffers.X_FLOAT_DATA)
        feature_extractor_step = matrix_features.LinearFloat32MatrixFeatureExtractorStep_f32i32(feature, feature_ordering)
    # elif data_type == 'depth_image':
 
    else:
        raise Exception("unknown data_type")

    node_steps.extend([feature_params_step, feature_extractor_step])

    slice_weights_step = pipeline.SliceFloat32VectorBufferStep_i32(sample_data_step.WeightsBufferId, sample_data_step.IndicesBufferId)
    node_steps.append(slice_weights_step)
    if prediction_type == 'classification':
        slice_ys_step = pipeline.SliceInt32VectorBufferStep_i32(buffers.CLASS_LABELS, sample_data_step.IndicesBufferId)
        number_of_classes = int( np.max(kwargs['classes']) + 1 )
        if split_type == 'best_gap_midpoint':
            class_infogain_walker = classification.ClassInfoGainWalker_f32i32(slice_weights_step.SlicedBufferId,
                                                                              slice_ys_step.SlicedBufferId,
                                                                              number_of_classes)
            best_splitpint_step = classification.ClassInfoGainBestSplitpointsWalkingSortedStep_f32i32(class_infogain_walker,
                                                                                feature_extractor_step.FeatureValuesBufferId,
                                                                                feature_ordering)
            node_steps.extend([slice_ys_step, best_splitpint_step])
        else:
            raise Exception("unknown split_type")
        finalizer = classification.ClassEstimatorFinalizer_f32()

        y_estimator_dimension = number_of_classes
    elif prediction_type == 'regression':
        slice_ys_step = pipeline.SliceFloat32MatrixBufferStep_i32(buffers.YS, sample_data_step.IndicesBufferId)
        raise Exception("not implemented")
    else:
        raise Exception("unknown data_type")


    split_buffers = splitpoints.SplitSelectorBuffers(best_splitpint_step.ImpurityBufferId,
                                                          best_splitpint_step.SplitpointBufferId,
                                                          best_splitpint_step.SplitpointCountsBufferId,
                                                          best_splitpint_step.ChildCountsBufferId,
                                                          best_splitpint_step.LeftYsBufferId,
                                                          best_splitpint_step.RightYsBufferId,
                                                          feature_params_step.FloatParamsBufferId,
                                                          feature_params_step.IntParamsBufferId,
                                                          feature_extractor_step.FeatureValuesBufferId,
                                                          feature_ordering,
                                                          feature_extractor_step)

    tree_steps_pipeline = pipeline.Pipeline(tree_steps)
    node_steps_pipeline = pipeline.Pipeline(node_steps)

    try_split_criteria = create_try_split_criteria(**kwargs)
    should_split_criteria = create_should_split_criteria(**kwargs)

    split_indices = splitpoints.SplitIndices_f32i32(sample_data_step.IndicesBufferId)
    split_selector = splitpoints.SplitSelector_f32i32([split_buffers], should_split_criteria, finalizer, split_indices)

    if tree_type == 'depth_first':
        tree_learner = learn.DepthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    elif tree_type == 'breadth_first':
        tree_learner = learn.BreadthFirstTreeLearner_f32i32(try_split_criteria, tree_steps_pipeline, node_steps_pipeline, split_selector)
    else:
        raise Exception("unknown tree_type")

    number_of_trees = int( kwargs.get('number_of_trees', 10) )
    number_of_jobs = int( kwargs.get('number_of_jobs', 1) )
    forest_learner = learn.ParallelForestLearner(tree_learner, number_of_trees, y_estimator_dimension, number_of_jobs)
    return forest_learner


def create_uber_learner(**kwargs):
    return LearnerWrapper(  make_uber_data_prepare(kwargs),
                            uber_create_learner,
                            make_uber_create_predictor(kwargs),
                            kwargs)
