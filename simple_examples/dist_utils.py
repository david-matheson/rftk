import numpy as np
import matplotlib.pyplot as plt
import random

from math import sqrt, pi

def gaussian_pdf(mu, sigma, x):
    dim = len(mu)
    d = 1. / (((2*pi)**(dim/2.0))*sqrt(np.linalg.det(sigma))) * np.exp(-0.5*np.dot((x-mu), np.dot(sigma, (x-mu)).T))
    return float(d)

class MixtureOfGaussians:
    def __init__(self, mixture, mus, covs):
        self.mixture = mixture
        self.mus = mus
        self.covs = covs
        self.number_classes = len(mixture)
        self.x_dim = len(mus[0])

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
                y_probs[i,c] = self.mixture[c] * gaussian_pdf(self.mus[c], self.covs[c], xs[i])
            y_probs[i,:] = y_probs[i,:] / np.sum(y_probs[i])
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

