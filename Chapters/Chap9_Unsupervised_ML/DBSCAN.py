import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

if True:
    X, y = make_moons(n_samples=1000, noise=0.05)
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X)
    print(dbscan.labels_)       # If =-1 -> anomaly
    # core instances:
    print(dbscan.core_sample_indices_)
    # Coordinates in the feature space of each core instances
    print(dbscan.components_)

    # To predict new instances, we could use KNN trained on the core instances
    # We could train on all the instances but the anomalies if we wanted
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

    X_new = np.array([ [-0.5, 0], [0, 0.5], [1, -0.1], [2, 1] ])
    print(knn.predict(X_new))
    print(knn.predict_proba(X_new))

    # To find anomalies (points too far from clusters)
    y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)   # returns the distances and idx to the k nearest
                                                                # instances

    y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
    y_pred[y_dist > 0.2] = -1
    print(y_pred.ravel())
