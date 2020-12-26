# Linear SVM
#
#   Useful for linear and non-linear classification, regression and outlier detection
#
#   Well suited for complex small or medium sized datasets
#
#   Separates as well as posible the classes (so that they are all as far as possible from the separation boundary)
#   using a hypersurface in the parameter space
#
#   -> fits "the best street" between the data
#
#   The instances at the edge are the *support vector*
#
#   Recommended to feature scale (e.g.\ sklearn StandardScaler())


# Hard-margin classification
#   -> if we impose that all instances must be on the right side of the decision boundary
#  Probs:
#   -> only work if data linearly separable
#   -> Sensitive to outliers
#
# Soft margin classification:
#
#   => Use a more flexible model!:
#       - use as wide a gap between the classes with as few instances as possible on the wrong side of the decision
#       boundary


# SVM in scikit-learn takes many hyperparameters:
#
#   C: low value: wide margin with many data within it
#      large val: narrow margin with few points within it
#
#   if SVM overfits -> can try to regularize by reducing C
#
#
# LinearSVC regularizes the bias term!
#   -> Should center the training set first by removing the mean
#   or should use StandardScaler

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC   # Support Vector Classification


def plot_predictions(model, X, y):
    import matplotlib.pyplot as plt
    # Look at the probabilities given a petal width between 0 och 3 cm
    dx, dy = 0.05, 0.05
    x_min, x_max = 2, 8
    y_min, y_max = 0, 3
    y_grid, x_grid = np.mgrid[y_min:y_max+dy:dy, x_min:x_max+dx:dx]
    X_new = np.concatenate((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), axis=1)
    y_class = model.predict(X_new)
    plt.pcolor(x_grid, y_grid, y_class.reshape(y_grid.shape[0], y_grid.shape[1]), label="Probability of Iris virginica")
    plt.colorbar()

    y1 = y == 1
    y0 = y == 0

    plt.scatter(X[y1, 0], X[y1, 1], c='r', marker='^', label="Iris virginica")
    plt.scatter(X[y0, 0], X[y0, 1], c='c', marker='o', label="Not iris virginica")

    plt.xlim(2, 8)
    plt.ylim(0, 3)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()


def load_data():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = (iris["target"] == 2).astype(np.float64)
    return X, y


if False:
    X, y = load_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
        ])
    model.fit(X, y)
    # SVM classifiers do not output probabilities for each class
    # just the classification
    plot_predictions(model, X, y)

if False:
    from sklearn.svm import SVC
    X, y = load_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", SVC(kernel='rbf', C=1)),
        ])
    model.fit(X, y)
    plot_predictions(model, X, y)

if False:
    from sklearn.linear_model import SGDClassifier
    X, y = load_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        # Same as linear SVM but allows to train on larger data sets that do not fit in the memory
        ("SGDClassifier", SGDClassifier(loss="hinge", alpha=1 / 1000)),
        ])
    model.fit(X, y)
    plot_predictions(model, X, y)
