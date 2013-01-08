import numpy as np

def max_of_n_prediction(Y_prob):
    """
    Y_prob: A size (n_data, n) numpy matrix where Y_prob[i,j] is proportional to
    the probability that data point i belongs to class j.

    Returns: A size (n_data,) numpy vector Y where Y[i] is the most likely class
    for data point i, based on Y_prob.
    """
    return Y_prob.argmax(axis=1)


def max_prob_of_n_prediction(Y_prob):
    """
    Y_prob: A size (n_data, n) numpy matrix where Y_prob[i,j] is proportional to
    the probability that data point i belongs to class j.

    Returns: A size (n_data,) numpy vector Y where Y[i] is the probability of the
    most likely class for data point i, based on Y_prob.
    """
    return Y_prob.max(axis=1)


def colors_from_predictions(Y_hat, colors):
    """
    Y_hat: A length n numpy array where Y_hat[i] is the class with the highest
    probability
    colors: List of rgb values to assign for each class

    Returns: A size (n_data,3) numpy vector y_colors where y_colors[i] is the color
    of the class Y_hat[i]
    """
    y_colors = np.zeros((len(Y_hat),3))
    for yi in range(len(colors)):
        y_colors[ Y_hat == yi ] = colors[yi]
    return y_colors


def image_from_predictions(Y_hat, Y_probs, colors, shape):
    """
    Y_hat: A length n numpy array where Y_hat[i] is the class with the highest
    probability
    Y_probs:  A length n numpy array where Y_probs[i] is the probability of Y_hat[i]
    colors: List of rgb values to assign for each class
    shape: 2 dimension shape of output image

    Returns: A size (shape[0],shape[1],3) numpy vector img where img[i,j] is
    the color of the class Y_hat[i*shape[1] + j] and shaded by the probability
    """
    img_flat = colors_from_predictions(Y_hat, colors)
    # darken colors by probability
    img_flat = (img_flat.T * Y_probs.T * Y_probs.T).T

    img = img_flat.reshape((shape[0],shape[1],3))
    return img