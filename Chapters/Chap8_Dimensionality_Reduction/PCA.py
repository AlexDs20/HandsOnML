# Principal Component Analysis
#   PCA assumes that the data are centered around the origin!!
#   Scikit-learn does the centering for you
# : X = U SIGMA V*
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def load_data():
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))
    X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
    return X

# Doing PCA using numpy
if 0:
    X = load_data()
    X_centered = X-X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered)
    c1 = Vt.T[:, 0]
    c2 = Vt.T[:, 1]
    # Project onto the ifrst two principal components:
    W2 = Vt.T[:, :2]
    X2d = X_centered.dot(W2)

def plot_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0],X[:, 1], X[:, 2], linewidths=3, alpha=None, marker='.')
    plt.show()

# Using Scikit-Learn
# It centers the data automatically!
if 0:
    X = load_data()
    pca = PCA(n_components = 2)
    X2D = pca.fit_transform(X)
    print(pca.components_)      # Contains the basis vector of the subspace

    # explained variance ratio
    print(pca.explained_variance_)  # proportion of dataset's variance along those axis

# Number of dimensions?
if 1:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.uint8)
    from sklearn.model_selection import train_test_split
    X = mnist["data"]
    y = mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

if 0:
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum>=0.95) + 1
    print(d)
    pca = PCA(n_components = d)     # Should be part of the pipeline
    X_reduced = pca.fit_transform(X_train)
    # We can "get back the data" (with a bit of loss of course) by doing the inverse transfo.
    X_recovered = pca.inverse_transform(X_reduced)

    # Alternatively, to get 95%:
if 0:
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_train)

    # Or plot the cumsum and see what works well (after the elbow)
if 0:
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum>=0.95) + 1
    def plot_cumsum():
        plt.figure(figsize=(10,6))
        plt.plot(cumsum, linewidth = 3)
        plt.axis([0, 400, 0, 1])
        plt.xlabel("Dimensions")
        plt.ylabel("Variance explained")
        plt.show()
    plot_cumsum()

if 0:
    rnd_pca = PCA(n_components=154, svd_solver="randomized")
    X_reduced = rnd_pca.fit_transform(X_train)

# Incremental PCA
if 0:
    from sklearn.decomposition import IncrementalPCA
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)

    X_reduced = inc_pca.transform(X_train)

# It's possible for numpy to only load the data that are needed:
# : memmap
if 0:
    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))

    batch_size = m // n_batches
    inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
    inc_pca.fit(X_mm)
