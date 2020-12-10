# Polynomial regression
#
# If non linear data? -> It can actually be modelled using linear model!
# how?: Add powers of each feature as new features: x2 := x1**2
# Train this on  using an extended set of features: *Polynomial Regression*
import numpy as np

def load_data():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
    return X, y

# Can use sci-kit learn PolynomialFeatures up to a max order:
# It creates all combinations of features to create higher-order polynomial as new features
# e.g. y = ax + bx^2  ->> y = ax + bz (linear) with z = x^2 and we try to get a and b
# Note that it takes all the combinations of the features! -> it increases quickly!
# y = ax + bx^2 + cz to 2nd order, Polynomial Features creates (given x and z): x^2 z^2 xz

if True:
    from sklearn.preprocessing import PolynomialFeatures
    X, y = load_data()
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)     # Contains the original and the higher order ones

    # Now can fit linear regression model to the extended dataset
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_poly, y)
    print(model.intercept_, model.coef_)
    x_check = -3
    print(model.predict([[x_check, x_check**2]]))

# Problem, which order should be used?
# If too low-order model -> underfitting
# If too high order -> overfitting
# Previously -> cross-validation to get an estimate of the model's generalization preformance
#       If good on training but bad on the cross-val -> over-fitting
#       If always bad -> underfitting
#
# Another way to check:
#       Look at the *learning curves*
#               -> Plots of the model's performance on the training set and
#                  the validation set as a function of the training set size (or iteration)
#       How?
#           -> Train the model several times on different size subsets of the training set

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.xlim((0, len(X_train)))
    plt.ylim((0, 4))
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.legend(['Train', 'Validation'])
    plt.show()

if False:
    X, y = load_data()
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    # Note we fit a second order by a linear model!
    plot_learning_curves(model, X, y)
    # Training error goes up because: data is noisy and because it's not linear!
    #       -> reach a plateau when adding a point doesn't make the error much better or worse
    # Validation error goes down because because at the beginning -> very bad prediction
    #       later, it learns and gets better but still bad model and plateau at same point as train error
    # Sign of underfitting:
    #       - both curves - reached a plateau
    #                     - close to each other
    #                     - fairly high

if True:
    X, y = load_data()
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    polynomial_regression = Pipeline([
            ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
            ("lin_reg", LinearRegression()),
            ])

    plot_learning_curves(polynomial_regression, X, y)

    # Similar to the linear fit but 2 main differences!!
    #   1) error on the training data is much lower than with the linear regression model
    #   2) Works much better on the training data than on the validation data -> overfitting model
    # To improve an overfitting model -> feed more training data until the validation error reaches the training error


# BIAS/VARIANCE TRADE-OFF
#
#   -> Generalization error is a sum of trhee errors:
#
#   1) Bias:
#       Wrong assumptions (wrong model: e.g. linear instead of quad)
#       a high-bias model is likely to underfit the training data
#
#   2) Variance:
#       Due to the model's large sensitivity to small variations in the training data
#       E.g. model with many degrees of freedom could have high varience due to overfit
#
#   3) Irreducible error:
#       Due to the noise in the data
#       To get rid of this: Clean the data! (e.g. detect and remove outliers)
#
# -> increased model complexity: increase varience and decrease the bias
