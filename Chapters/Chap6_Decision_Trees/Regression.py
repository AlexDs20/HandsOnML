import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

def load_data():
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 1)
    y = 4 * (X - 0.5) ** 2
    y = y + np.random.randn(m, 1) / 10
    return X, y

def visualise():
    DIR = 'Chapters/Chap6_Decision_Trees/'
    file_name = 'regression'
    ext = 'ps'      # or png
    export_graphviz(
            model,
            out_file=DIR+file_name+".dot",
            feature_names=["x1"],
            rounded=True,
            filled=True
        )
    os.system(f'dot -T{ext} {DIR}{file_name}.dot -o {DIR}{file_name}.{ext}')
    os.system(f'qpdfview {DIR}{file_name}.{ext}')

if True:
    X, y = load_data()
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    # visualise()
