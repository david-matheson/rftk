import numpy as np

class AccuracyMeasurement(object):
    MEASURED_VALUES = [
            'accuracy',
            ]

    def __init__(
            self,
            accuracy=-1,
            ):
        self.accuracy = accuracy



class StatsMeasurement(object):
    MEASURED_VALUES = [
            'accuracy',
            'min_depth',
            'max_depth',
            'average_depth',
            'min_estimator_points',
            'max_estimator_points',
            'average_estimator_points',
            'total_estimator_points',
            ]

    def __init__(
            self,
            accuracy=-1,
            min_depth=-1,
            max_depth=-1,
            average_depth=-1,
            min_estimator_points=-1,
            max_estimator_points=-1,
            average_estimator_points=-1,
            total_estimator_points=-1,
            ):
        self.accuracy = accuracy
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.average_depth = average_depth
        self.min_estimator_points = min_estimator_points
        self.max_estimator_points = max_estimator_points
        self.average_estimator_points = average_estimator_points
        self.total_estimator_points = total_estimator_points


