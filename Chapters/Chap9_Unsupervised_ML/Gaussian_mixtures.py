# Given a dataset, we kan get the weight of the the different gaussians as well as their parameters
import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

def load_data():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

def print_(gm):
    print('Weights', gm.weights_)
    print('Means',  gm.means_)
    print('Cov', gm.covariances_)
    print('Converged:', gm.converged_)
    print('n iter:', gm.n_iter_)



gm = GaussianMixture(n_components=3, n_init=10)
X, _ = load_data()
gm.fit(X)

print_(gm)

# Now the model can classify (hard clustering) or give a probability to belong to a cluster (soft clustering)

if 0:
    print(gm.predict(X))
    print(gm.predict_proba(X))

# Can generate new instances from Gaussian mixture model (generative model)

X_new, y_new = gm.sample(6)
print(X_new)
print(y_new)

# Estimate density of the model at any given location
# For each instance, it estimates the log of the probability density function -> the higher the score, the higher the
# density
print(gm.score_samples(X[1:10]))


# Anomaly detection using Gaussian Mixtures
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]


# How to find the number of clusters?
# Find the model that minimizes a theretical information criterion
print(gm.bic(X[0:10]))
print(gm.aic(X[0:10]))


#----------------------------------------
# Bayesian Gaussian Mixture Model
#
# Find the number of clusters automatically but requires an upper limit

from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(X)
print(np.round(bgm.weights_, 2))
