import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_data():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

def load_blobs():
    from sklearn.datasets import make_blobs
    blob_centers = np.array(
        [[ 0.2,  2.3],
         [-1.5 ,  2.3],
         [-2.8,  1.8],
         [-2.8,  2.8],
         [-2.8,  1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
    return X, y


#-----------
#  K-means
#-----------
if False:
    from sklearn.cluster import KMeans
    k = 5
    kmeans = KMeans(n_clusters=k)

    X, y = load_data()
    y_pred = kmeans.fit_predict(X)
    print(y_pred is kmeans.labels_)

    # Can find the centroids:
    kmeans.cluster_centers_

    # Can predic new instances by searching the closest centroid
    X_new  = np.array([ [0, 2], [-1, 2], [-3, 3], [-3, 2], [-3, 1] ])
    print(kmeans.predict(X_new))

    # Measures distance to each centroid for every instance
    kmeans.transform(X_new)

# with init
if False:
    from sklearn.cluster import KMeans
    good_init = np.array([ [-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2] ])
    k = 5

    kmeans = KMeans(n_clusters=k, init=good_init, n_init=1)

    X, y = load_data()
    y_pred = kmeans.fit_predict(X)
    print(y_pred is kmeans.labels_)

    # Can find the centroids:
    kmeans.cluster_centers_

    # Can predic new instances by searching the closest centroid
    X_new  = np.array([ [0, 2], [-1, 2], [-3, 3], [-3, 2], [-3, 1] ])
    print(kmeans.predict(X_new))

    # Measures distance to each centroid for every instance
    kmeans.transform(X_new)

# by default KMeans fits 10 models using different random centroid
# and keeps the best model
# Which one is best? -> inertia metric: mean squared distance between each instance and closest centroid
# -> standard KMeans method in sklearn

# using mini-batches
if False:
    from sklearn.cluster import MiniBatchKMeans
    minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
    X, y = load_data()
    minibatch_kmeans.fit(X)

# Use silhouette score for finding number of cluster
if False:
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    X, y = load_blobs()
    kmeans_per_k = [KMeans(n_clusters=k).fit(X) for k in range(1, 10)]
    silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

    plt.figure()
    plt.plot(range(2, 10), silhouette_scores)
    plt.show()

# Image Segmentation
if False:
    import os
    from matplotlib.image import imread
    from sklearn.cluster import KMeans

    image = imread(os.path.join("/home/alexandre/Pictures", "IMG-20190502-WA0004.jpg"))
    plt.figure()
    plt.imshow(image)
    plt.show()
    X = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=8).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)

    plt.figure()
    plt.imshow(segmented_img/255)
    plt.show()

# Preprocessing MNIST
if False:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GridSearchCV

    # Without preprocessing
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    # With preprocessing
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=50, random_state=42)),
        ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42))
        ])
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))

    # Finding the best number of clusters:
    para_grid = dict(kmeans__n_clusters=range(95, 100))
    grid = GridSearchCV(pipeline, para_grid, cv=3, verbose=2)
    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.scort(X_test, y_test))

# Use for semi-supervised learning
if True:
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train a model of 50 clusters on all the data
    # The found centroids are reference images for the classification
    k = 50
    model = KMeans(n_clusters=k, random_state=42)
    X_digits_dist = model.fit_transform(X_train)
    print(X_digits_dist.shape)

    # We choose the reference images for each centroid (which we label by hand) as the one closest to the centroid
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)
    print(representative_digit_idx)
    X_representative_digits = X_train[representative_digit_idx]

    # Plot the reference images
    plt.figure()
    for index, X_representative_digit in enumerate(X_representative_digits):
        plt.subplot(k // 10, 10, index + 1)
        plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
        plt.axis('off')
    plt.show()

    # which we now label by hand:
    y_representative_digits = np.array([
        4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
        5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
        1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
        6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
        4, 2, 9, 4, 7, 6, 2, 3, 1, 1])

    # Train the new model using the reference images as training set
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_representative_digits, y_representative_digits)
    print(log_reg.score(X_test, y_test))

    # To make it even better, we can propagate the labels of the reference images of each cluster to all the images in
    # the cluster
    y_train_propagated = np.empty(len(X_train), dtype=np.int32)
    for i in range(k):
        y_train_propagated[model.labels_==i] = y_representative_digits[i]

    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train_propagated)
    print(log_reg.score(X_test, y_test))

    # Now the labels are on all the images in the cluster which contain outliers.
    # It may be better to label only the images closest to the centroid
    percentile_closest = 20

    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), model.labels_]
    for i in range(k):
        in_cluster = (model.labels_ == i)
        cluster_dist = X_cluster_dist[in_cluster]
        cutoff_distance = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = (X_cluster_dist > cutoff_distance)
        X_cluster_dist[in_cluster & above_cutoff] = -1

    partially_propagated = (X_cluster_dist != -1)
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train_propagated[partially_propagated]

    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

    print(log_reg.score(X_test, y_test))
