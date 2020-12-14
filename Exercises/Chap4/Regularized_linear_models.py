# To reduce overfitting -> regularize the model (= use constraints)
#
# Simple way to regularized polunomial model -> reduce degree of polynomial
#
# Linear model -> constrain the weights of the model: Ridge Regression, Lasso Regression, Elastic Net

import numpy as np

def load_data():
    n = 100
    X = 2 * np.random.rand(n, 1)
    y = 4 + 3 * X + np.random.randn(n, 1)
    return X, y


# Ridge Regression:
#   Regularized version of linear regression with + alpha \sum_i=1^n \theta_i^2  (added to the cost function, alpha tells how much to regularize the model)
#       -> forces the model to learn the theta_i and keep them small
#   - the regularization term used only during training
#   - use unregularized cost function to evaluate the model's performances
#
#
#   Cost function for training and testing are often different:
#       - for optimization (training), it should have easy derivatives
#       - for performance: it should be as close as possible to "the best"
#                           (classifier use log loss for train but precision/recall for evaluation)
#
#   J(\theta) = MSE(\theta) + alpha/2 \sum_i=1^n \theta_i^2
#   (theta_0) is not regularized
#
#   for Gradient descent -> add \alpha*(\theta_1,...,\theta_n)
#
#   IMPORTANT: SCALE THE DATA USING StandardScaler before using regularized models
#
#   Ridge Regression can be solved in closed form or using gradient descent
#
#   Closed: forme solution:
#       \hat{\theta} = (X^tX + \alpha A)^-1 X^t y
#   A = I_{n+1 x n+1} with A[0, 0] = 0

if True:
    X, y = load_data()

if False:
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1, solver="cholesky")
    model.fit(X, y)
    print(model.predict([[1.5]]))

if False:
    from sklearn.linear_model import SGDRegressor
    model = SGDRegressor(penalty="l2")
    model.fit(X, y.ravel())
    print(model.predict([[1.5]]))

# Lasso Regression
#
# Similar to Ridge Regression but uses L1 norm instead of L2 norm
# J(\theta) = MSE(\theta) + \alpha \sum_i=1^n \abs{\theta_i}
#
# Tends to suppress the weights of the less significant features
#   (= automatic feature selection)
#
# With Lasso it will bounce around the minimum because of the \abs
#   -> to avoid it: reduce the learning rate during training

if False:
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.1)
    model.fit(X, y)
    print(model.predict([[1.5]]))

if False:
    from sklearn.linear_model import SGDRegressor
    model = SGDRegressor(penalty="l1")
    model.fit(X, y.ravel())
    print(model.predict([[1.5]]))


# Elastic Net:
#
# Middle between Ridge and Lasso:
#       -> regularization term is a mix of both controled by a ration *r*
#    r = 0 -> Ridge
#    r = 1 -> Lasso

if False:
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=0.1)
    model.fit(X, y)
    print(model.predict([[1.5]]))


# When to use Linear Regression, Ridge, Lasso, ElasticNet?
#
# Almost preferable to have a bit of regularization
#       -> Avoid plain Linear Regression
# Ridge is good default
# Prefer lasso or elastic net if suspect that only few features are important
# In general ElasticNet over Lasso


# Another way to regularize is to *EARLY STOP* the training
#   -> so that it does not have time to overfit
#
# When to stop?
#   The validation error will go down at the beginning.
#   When it starts going up again -> sign of overfitting -> stop training

if False:
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.base import clone

    import matplotlib.pyplot as plt

    # Prepare the data
    poly_scaler = Pipeline([
            ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
            ("std_scaler", StandardScaler())
            ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.fit_transform(X_val)

    model = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                        penalty=None, learning_rate="constant", eta0=5*10**-4)

    min_val_error = float("inf")
    best_epoch = None
    best_model = None

    epochs = None
    ERR = []

    for epoch in range(1000):
        model.fit(X_train_poly_scaled, y_train.ravel())     # Continues from previous (because warm_start = True)
        y_val_predict = model.predict(X_val_poly_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)
        ERR.append(val_error)
        if val_error < min_val_error:
            min_val_error = val_error
            best_epoch = epoch
            best_model = clone(model)

    plt.plot(range(1000), ERR)
    plt.title(best_epoch)
    plt.show()
