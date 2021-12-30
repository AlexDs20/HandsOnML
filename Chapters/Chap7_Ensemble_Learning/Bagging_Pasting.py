from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

# By default, BaggingClassifier does soft voting if possible (if \verb;predict_proba(); exists.)
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))

# To use the out-of-bag samples for evaluation: oob_score=True
bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit(X_train, y_train)
# Gives an idea for the accuracy on the test set:
bag_clf.oob_score_
y_pred = bag_clf.predict(X_test)
print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred))
# The dedicision function for the oob instances:
bag_clf.oob_decision_function_

# Random Patches and Random Subspaces
#
#   BaggingClassifier -> can also sample the features
#                     -> 2 hyperparameters: max_features and bootstrap_features
#
#   ExtraTreesRegressor similar API to RandomForestRegressor
