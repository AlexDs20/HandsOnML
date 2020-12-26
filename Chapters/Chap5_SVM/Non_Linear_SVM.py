# Non Linear SVM
#
#   In many cases the data are not linearly separable
#
#   -> Add more features (e.g.\ x2 = (x1**2))

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC   # Support Vector Classification
from sklearn.svm import SVC
import numpy as np


def plot_predictions(model, X, y):
    import matplotlib.pyplot as plt
    # Look at the probabilities given a petal width between 0 och 3 cm
    feat_1 = 0
    feat_2 = 1
    dx, dy = 0.05, 0.05
    x_min, x_max = -1.5, 2.5
    y_min, y_max = -1.0, 1.5

    y_grid, x_grid = np.mgrid[y_min:y_max+dy:dy, x_min:x_max+dx:dx]
    X_new = np.concatenate((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), axis=1)
    y_class = model.predict(X_new)
    plt.pcolor(x_grid, y_grid, y_class.reshape(y_grid.shape[0], y_grid.shape[1]),
               label="Class Prediction", cmap=plt.get_cmap("jet"))
    plt.colorbar()

    y1 = y == 1
    y0 = y == 0

    plt.scatter(X[y1, feat_1], X[y1, feat_2], c='r', marker='^', label="True")
    plt.scatter(X[y0, feat_1], X[y0, feat_2], c='c', marker='o', label="False")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(f'Feature {feat_1}')
    plt.ylabel(f'Feature {feat_2}')
    plt.legend()
    plt.show()


if False:
    X, y = make_moons(n_samples=100, noise=0.15)
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
        ])
    model.fit(X, y)
    plot_predictions(model, X, y)


# Polynomial Kernel
#
#   Adding polynomial features is simple and works great with many machine learning algorithms
#
#   For complex data -> it would require high polynomial degree -> many features -> make the model slow
#
#   Instead:
#       -> use the kernel trick:
#               Get same results as if many polunomial features wihtou having to add them -> no speed problem
#
#   SVC:
#    can adjust the degree if under/over-fitting
#    The coef0 hyperparameter controls how much the model is influenced by high vs low-degree polynomials
#    -> use coarse gridsearch first and the more precise grid search near the regions of the hyperparameters space that
#    can be of interest

if False:
    X, y = make_moons(n_samples=100, noise=0.15)
    model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
                ])
    model.fit(X, y)
    plot_predictions(model, X, y)


# Similarity features
#
#   another way to solve non-linear problems:
#
#       Add Features using *similarity function*: it measures how much an instance resembles a *landmark*
#
#       How to select the landmark?
#           One at each instance in the dataset
#           -> creates as many features as there are instances!

# This hsould be implemented to be Pipeline complient i.e.\ create a class with fit and transform methods
def RBF(x, landmark=0):
    gamma = 0.3
    return np.exp(-gamma * np.square(x-landmark))


if False:
    X, y = make_moons(n_samples=100, noise=0.15)

    new_feat = RBF(X[:, 1], 0)
    X = np.c_[X, new_feat]


# Gaussian RBF (Radial Basis Function) Kernel
#
#   Again creating features using similarity features is expensive computationally
#
#       -> kernel trick
#
# Increasing gamma makes the bell-shaped curve narrower
#       -> decision boundary is more irregular and wiggling around individual instances
#
# Decreasing gamma makes the bell-shaped curve wider
#       -> decision boundary is smoother
#
#   => Gamma acts like a regularization hyperparameter
#       - if over-fitting: reduce gamma
#       - if under-fitting: increase gamma

if True:
    X, y = make_moons(n_samples=100, noise=0.15)
    model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
            ])
    model.fit(X, y)
    plot_predictions(model, X, y)


# Other kernels exist but are used more rarely
#   e.g. string kernels: to classify text documents or DNA sequences
#
#
#   Which kernel to use?
#
#       Start with LinearSVC (much faster than SVC(kernel="linear"))
#       If training set is not too large, also try Gaussian RBF kernel
#
#   Use corss-validation and grid-search
