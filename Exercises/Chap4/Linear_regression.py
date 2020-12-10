# Linear models:
#   Say you have x_1 ... x_n features, a linear model predicts:
#       yhat = th0 + th1*x1 + ... + th_n*x_n
#
#   The constant: theta0 is the bias (also called the intercept term)
#
# Written in vector notations:
# yhat = h_{\theta}(x) = \theta \dot x
#
# h is the "hypothesis function"
#
# -> training the model = Fitting the parameters that best fit the training set
# -> Need a measure of how good or bad the fit is: RMSE
# -> Need to find theta that minimizes RMSE. We actually do it on MSE because simpler and same result.
#
#   MSE(X, h_theta) = 1/m * (theta_j x_j^i - y^i)^2
#
#   The normal equation (solution to the minimum of MSE):
#       \hat\theta = (X^tX)^-1 X^t y

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Simple linear example and finding analytical solution and checking how it does
if True:
    def load_data():
        n = 100
        X = 2 * np.random.rand(n, 1)
        y = 4 + 3 * X + np.random.randn(n, 1)
        return X, y

if False:
    if True:
        X, y = load_data()

        X_b = np.c_[np.ones((100,1)), X]        # Adds x0=1 to each instance
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        print(theta_best)

        X_new = np.array([[0], [2]])
        X_new_b = np.c_[np.ones((2, 1)), X_new]
        y_predict = X_new_b.dot(theta_best)
        print(y_predict)

        plt.plot(X, y, 'b.')
        plt.plot(X_new, y_predict, 'r-')
        plt.axis([0, 2, 0, 15])
        plt.show()

    if True:
        X, y = load_data()
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        lin_reg.intercept_, lin_reg.coef_
        X_new = np.array([[0], [2]])
        lin_reg.predict(X_new)
        # Here the solution is calculated using the pseudo inverse and using singular value decomposition

# The two previous methods to solve the equation for theta are very slow when the number of features increases
# because you need to invert the matrix which complexity gets larger and larger
# When there are large amount of features or too many training instances (to fit memory)

# Implement gradient descent:
# Adjust iteratively the parameters theta to minimize a cost function
# *Gradient descent*: takes local gradient wrt theta and take a step in that direction and update the theta
# *Learning Rate* : controls the size of the step in the direction of the gradient.
#                   It cannot be too small (takes long to converge), or too big (jump around the minimum)
#
# If several features that must be optimized:
#       If they have different value range, a small step is needed for one feature (theta1) for it to converge
#       but it will take a long time to converge for the second feature (theta2) as the learning rate (eta) must be small (for theta1)
#       but we must change by a lot the value of theta2
#
# -> When using gradient descent, be sure that the features have similar scale
#   e.g. use Scikitlearn StandardScaler()
#
# \partial_j Cost_Function

def Gradient_Descent_MSE(x, y, theta, learning_rate, tolerence, max_it=10**4):
    x_new = np.c_[np.ones((len(x),1)), x]
    m = len(y)
    it = 0

    Grad = (2/m) * x_new.T.dot(x_new.dot(theta) - y)
    while (np.linalg.norm(Grad) > tolerence) and it < max_it :
        it = it + 1
        Grad = (2/m) * x_new.T.dot(x_new.dot(theta) - y)
        theta = theta - learning_rate * Grad
    return theta

if False:
    X, y = load_data()
    theta = np.random.randn(2,1)
    lr = 0.05
    tolerence = 0.01
    theta = Gradient_Descent_MSE(X, y, theta, lr, tolerence, max_it=10**3)
    print(theta)

    # Note that the results depend clearly on the learning rate!

# How many steps?
#    Stat with a lot but stop when the gradient is below a threshold (known as tolerence: \epsilon)
#
# Note that the problem is that it needs all the data to calculate the gradient with this method
#   For a large dataset -> this is really slow!
#       --> use a subset of instances for each step when calculating the gradient
#
# Stochastic Gradient:
#       - uses one random instance at every steps of the calculation of theta
#       - Good: Can train on huge sets and is fast, can also jump out of local minima
#       - Bad: Less regular and does not stabalize at the minimum but fluctuates around
#       - One solution is to reduce the learning rate: *learning schedule*
#               - If learning rate reduced too quickly -> get stuck in local minimum or stuck half way to the minimum
#               - If learning rate reduced too slowly -> jump around the minimum for long and get sub-optimal solution if stopped too early
def learning_schedule(t0, t1, t):
    return t0/ (t + t1)

def Stochastic_Gradient_Descent(theta, x, y):
    n_epochs = 100
    t0, t1 = 5, 50      # learning schedule hyperparameters
    m = 50
    x_new = np.c_[np.ones((len(x),1)), x]

    for epoch in range(n_epochs):
        for i in range(m):
            rand_idx = np.random.randint(m)
            xi = x_new[rand_idx:rand_idx+1]
            yi = y[rand_idx:rand_idx+1]
            grad = 2* xi.T.dot(xi.dot(theta) - yi)
            lr = learning_schedule(t0, t1, epoch * m + i)
            theta = theta - lr * grad
    return theta

def Stochastic_Gradient_Descent(theta, x, y):
    n_epochs = 50
    t0, t1 = 5, 50      # learning schedule hyperparameters
    m = len(y)
    x_new = np.c_[np.ones((len(x),1)), x]

    for epoch in range(n_epochs):
        for i in range(m):
            rand_idx = np.random.randint(m)
            xi = x_new[rand_idx:rand_idx+1]
            yi = y[rand_idx:rand_idx+1]
            grad = 2* xi.T.dot(xi.dot(theta) - yi)
            lr = learning_schedule(t0, t1, epoch * m + i)
            theta = theta - lr * grad
    return theta

# Each round of m interations is an epoch
# Note that we may miss some instances and others may be used several times
# Alternatively, at each loop, you could shuffle the instances (and the labels jointly)
# and go through them one by one


if False:
    X, y = load_data()
    theta = np.random.randn(2,1)

    theta = Stochastic_Gradient_Descent(theta, X, y)
    print(theta)

# When using SGD: the training instances must be independt and identically distributed
# to converge towards the global minimum
# To ensure this:
#   -> shuffle the instances during training:
#               - pick each instances randomly
#               - or shuffle the training set at the beginning of each epoch
# If no shuffle and the data are sorted by label -> then sgd will optimize on label at a time

# Using scikit-learn for linear regression: SGDRegressor

if True:
    from sklearn.linear_model import SGDRegressor
    X, y = load_data()
    model = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    model.fit(X, y.ravel())
    print(model.intercept_, model.coef_)

# Mini-batch Gradien Descent:
#   At each step, computes the gradients on small random sets of instances (called mini-batches)
#   (Advantage over SGD: performance boost from hardwar opt.)
#   It is less eratic than SGD
#           -> is closer to the minimum than SGDRegressor
#          -> harder to escape from a local minima
# In the case of linear regression, only batch GD stops at the minimum while SGD
# and mini-batch move around the minimum
#
# For performances for linear regression: page 188
