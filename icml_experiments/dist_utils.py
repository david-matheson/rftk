import numpy as np
import matplotlib.pyplot as plt
import random

from math import sqrt, pi

def gaussian_pdf(mu, sigma_det, sigma_inv, x):
    dim = len(mu)
    d = 1. / (((2*pi)**(dim/2.0))*sqrt(sigma_det)) * np.exp(-0.5*np.dot((x-mu), np.dot(sigma_inv, (x-mu)).T))
    return float(d)

class MixtureOfGaussians:
    def __init__(self, mixture, mus, covs):
        self.mixture = mixture
        self.mus = mus
        self.covs = covs
        self.number_classes = len(mixture)
        self.x_dim = len(mus[0])
        self.covs_det = map(np.linalg.det, covs)
        self.covs_inv = map(np.linalg.inv, covs)

    def sample(self, n):
        # There must be a better way of doing this
        ys = np.zeros(n, dtype=np.int32)
        xs = np.zeros((n, self.x_dim), dtype=np.float32)
        for i in range(n):
            yi = np.random.multinomial(1, self.mixture).argmax()
            v = np.random.multivariate_normal(self.mus[yi], self.covs[yi])
            xs[i,:] = v
            ys[i] = yi
        return xs, ys

    def predict_proba(self, xs):
        # Again, there must be a better way of doing this
        (number_of_datapoints, _) = xs.shape
        y_probs = np.zeros((number_of_datapoints, self.number_classes))
        for i in range(number_of_datapoints):
            for c in range(self.number_classes):
                y_probs[i,c] = self.mixture[c] * gaussian_pdf(self.mus[c], self.covs_det[c], self.covs_inv[c], xs[i])
            y_probs[i,:] = y_probs[i,:] / np.sum(y_probs[i])
        return y_probs

    def prob(self, xs):
        # Again, there must be a better way of doing this
        (number_of_datapoints, _) = xs.shape
        y_probs = np.zeros((number_of_datapoints, self.number_classes))
        for i in range(number_of_datapoints):
            for c in range(self.number_classes):
                y_probs[i,c] = self.mixture[c] * gaussian_pdf(self.mus[c], self.covs_det[c], self.covs_inv[c], xs[i])
            y_probs[i,:] = y_probs[i,:]
        return y_probs

def mog_2d_3class_example1():
    mixture = np.array([0.33, 0.33, 0.34])
    mus = [np.array([3,1]), np.array([1.5,-2]), np.array([-1, 0.5])]
    covs = [np.eye(2), np.eye(2), np.eye(2)]
    return MixtureOfGaussians(mixture, mus, covs)

def mog_2d_3class_example2():
    mixture = np.array([0.01, 0.04, 0.95])
    mus = [np.array([3,1]), np.array([1.5,-2]), np.array([-1, 0.5])]
    covs = [np.eye(2), np.eye(2), np.eye(2)]
    return MixtureOfGaussians(mixture, mus, covs)

def mog_2d_5class_example1():
    mixture = np.array([0.20, 0.33, 0.30, 0.15, 0.02])
    mus = [np.array([2, 2]),    # r
            np.array([6, 2]),    # g
            np.array([4, 0]),    # b
            np.array([8, 0]),    # y
            np.array([5, 2])]   # w
    covs = [3.0 * np.array([[1, 0.5], [0.5, 1]]),
            3.0 * np.array([[1, -0.5], [-0.5, 1]]),
            3.0 * np.eye(2),
            3.0 * np.eye(2),
            0.1 * np.eye(2)]
    return MixtureOfGaussians(mixture, mus, covs)

def get_mog_dist(name):
    d = {'3-class': mog_2d_3class_example1(), '5-class':mog_2d_5class_example1()}
    return d[name]


def generate_multi_pass_dataset(X_train, Y_train, data_size, number_of_passes_through_data):
    (m,n) = X_train.shape
    all_row_indices = range(m)
    random.shuffle( all_row_indices )

    X_train = X_train[all_row_indices, :]
    Y_train = Y_train[all_row_indices]

    X_train = X_train[0:data_size,:]
    Y_train = Y_train[0:data_size]

    row_indices = range(data_size)
    random.shuffle( row_indices )
    X_train = X_train[row_indices, :]
    Y_train = Y_train[row_indices]
    for i in range(number_of_passes_through_data-1):
        random.shuffle( row_indices )
        X_train = np.append(X_train, X_train[row_indices, :], axis=0)
        Y_train = np.append(Y_train, Y_train[row_indices])
    return (X_train, Y_train)
