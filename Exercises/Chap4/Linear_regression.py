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

if True:
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

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
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_reg.intercept_, lin_reg.coef_
    X_new = np.array([[0], [2]])
    lin_reg.predict(X_new)
    # Here the solution is calculated using the pseudo inverse and using singular value decomposition

# The two previous methods to solve the equation for theta are very slow when the number of features increases
# because you need to invert the matrix which complexity gets larger and larger
# When there are large amount of features or too many training instances (to fit memory)
