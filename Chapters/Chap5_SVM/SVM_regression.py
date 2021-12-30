# SVM Regression
#
#   SVM supports linear and non-linear regression.
#
#   In this case, instead of trying to fit an iso-surface that best separates the classes,
#   we try to have a surface (with margins -> hyperparameter \epsilon) that best encompass the data.
#
#   What is within the range prediction +- \epsilon does not get any penalty associated with the training loss
#
#   Adding more training instances within the margins does not affect the model's prediction

from sklearn.svm import LinearSVR
import numpy as np


def load_data():
    n = 100
    X = 2 * np.random.rand(n, 1)
    y = 4 + 3 * X + np.random.randn(n, 1)
    return X, y


def plot_predictions(model, X, y):
    import matplotlib.pyplot as plt
    # Look at the probabilities given a petal width between 0 och 3 cm

    x_predict = model.predict(X)

    plt.plot(X, x_predict, 'b.')
    plt.plot(X, y, 'ro')

    x_min, x_max = -1, 3
    y_min, y_max = 2, 11

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('x')
    plt.ylabel('Regression')
    plt.show()


if True:
    X, y = load_data()
    model = LinearSVR(epsilon=1.5)
    model.fit(X, y.ravel())
    plot_predictions(model, X, y)


# Non-linear regression
#
#   Use kernelized SVM models
#

if True:
    from sklearn.svm import SVR
    model = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
    model.fit(X, y)
    plot_predictions(model, X, y)


# Computational efficiency:
#
#       Linear SVR and Linear SVC scale linearly with the training instances
#
#       SVR and SVC gets are much slower for larger training sets
