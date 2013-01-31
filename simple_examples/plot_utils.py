import numpy as np
import matplotlib.pyplot as plt

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


def grid_plot(predictor, X_train, Y_train, X_test, plot_filename, plot_scatter=False):
    grid_extend = [X_test[:,0].min(), X_test[:,0].max(), X_test[:,1].min(), X_test[:,1].max()]
    Ux, Uy = np.meshgrid(
            np.linspace(grid_extend[0], grid_extend[1]),
            np.linspace(grid_extend[2], grid_extend[3]),
        )

    X_grid = np.concatenate([
        Ux.reshape((-1,1)), Uy.reshape((-1,1))],
        axis=1)

    Y_probs = predictor.predict_proba(X_grid)
    Y_hat = Y_probs.argmax(axis=1)

    plt.figure()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1]])
    img = image_from_predictions(Y_hat, Y_probs.max(axis=1), colors, Ux.shape)
    plt.imshow(img, extent=grid_extend, origin='lower')
    if plot_scatter:
        plt.scatter(X_train[:,0], X_train[:,1], c=colors_from_predictions(Y_train, colors))
    plt.savefig(plot_filename)
    plt.close()