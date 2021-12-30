# Logistic regression
#
# -> Commonly used to estimate the probability for an instance to belong in a class


# Estimating Probabilities
#
# -> calculates the logistic of a weighted sum of the input features
#
#   -> p = \sigma(x^T\theta)
#
#   \sigma(t) = \frac{1}{1 + exp(-t)}
#   value between 0 and 1
#   \sigma(t<0) < 0.5
#   \sigma(t>0) > 0.5
#
# -> prediction: y = 0 if p<0.5
#                    1 if p>=0.5


# Training and Cost Function
#
# Goal:
#   Find \theta such that the probability is
#           high for instances with y=1
#      and  low for instances with y=0
#
#
# Cost Function for 1 instance:
#               - log(p) if y=1
#   c(\theta) = - log(1-p) if y=0
#
# On all the training instances -> average
#
#   J(\theta) = -(1/m) * sum_{i=1}^m [y^(i) log(p^(i)) + (1-y^(i)) log(1-p^(i)]
#
# -> no closed-form equation to get \tetha that minimizes J
# but hte function is convect
# -> minimum is global and we can use Gradient Descent
#
# \partial_j J(\theta) = \frac{1}{m} * Sum_{i=1}^m \left(\sigma\left(\theta^Tx^(i)\right) -y^(i)\right) x_j^(i)
#
# Once we have gradient, we can used batch GD, SGD or mini-batch GD


# Logistic regression on iris dataset:

# to detect Iris virginica based only on the petal with feature
if False:
    from sklearn import datasets
    import numpy as np
    iris = datasets.load_iris()
    list(iris.keys())
    print(iris["DESCR"])
    X = iris["data"][:, 3:]      # Petal width
    y = (iris["target"] == 2).astype(np.int)    # 1 if Iris virginica, else 0

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)

    import matplotlib.pyplot as plt
    # Look at the probabilities given a petal width between 0 och 3 cm
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = model.predict_proba(X_new)
    # y_proba = model.predict(X_new)
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
    plt.plot(X_new, y_proba[:, 0], "b-", label="Not Iris virginica")
    plt.plot(X[(y == 0)], y[(y == 0)], "b^")
    plt.plot(X[(y == 1)], y[(y == 1)], "g^")
    plt.xlabel('Petal Width')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


# to detect Iris virginica based on 2 features
if True:
    from sklearn import datasets
    import numpy as np
    iris = datasets.load_iris()
    X = iris["data"][:, 2:]
    y = (iris["target"] == 2).astype(np.int)    # 1 if Iris virginica, else 0

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)

    import matplotlib.pyplot as plt
    # Look at the probabilities given a petal width between 0 och 3 cm
    dx, dy = 0.1, 0.1
    y_grid, x_grid = np.mgrid[0:8+dy:dy, 0:8+dx:dx]
    X_new = np.concatenate((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), axis=1)
    y_proba = model.predict_proba(X_new)
    # y_class = model.predict(X_new)

    plt.pcolor(x_grid, y_grid, y_proba[:, 1].reshape(81, 81), label="Probability of Iris virginica")
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
