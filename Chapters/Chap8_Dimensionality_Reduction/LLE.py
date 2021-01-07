# Locally Linear Embedded
#
if 1:
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.datasets import make_swiss_roll

    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    X_reduced = lle.fit_transform(X)
