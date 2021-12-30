import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def load_data():
    iris = load_iris()
    X = iris.data[:, 2:]    # Petal length and width
    y = iris.target
    return X, y

if True:
    iris = load_iris()
    X = iris.data[:, 2:]    # Petal length and width
    y = iris.target
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X, y)

    # scikit-learn uses the CART algo -> only two categories per node
    # other algo. ID3 -> Decision trees with nodes with more than two children

# Visualize the tree !
def visualise():
    DIR = 'Chapters/Chap6_Decision_Trees/'
    file_name = 'iris_tree'
    ext = 'ps'      # or png
    export_graphviz(
            model,
            out_file=DIR+file_name+".dot",
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )
    os.system(f'dot -T{ext} {DIR}{file_name}.dot -o {DIR}{file_name}.{ext}')
    os.system(f'qpdfview {DIR}{file_name}.{ext}')

# Prediction that it belongs to a class:
if False:
    model.predict_proba([[5, 1.5]])
    model.predict([[5, 1.5]])

# Gini Impurity or Entropy
if True:
    iris = load_iris()
    X = iris.data[:, 2:]    # Petal length and width
    y = iris.target
    model = DecisionTreeClassifier(max_depth=2,
                criterion="entropy"
            )
    model.fit(X, y)
    # visualise();
