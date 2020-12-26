import os
import joblib
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
                            f1_score, precision_recall_curve, roc_curve, \
                            roc_auc_score


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier


def import_mnist(download=False):
    p = os.path.join('../data', 'mnist')
    if dl:
        mnist = fetch_openml('mnist_784', version=1)
        os.makedirs('../data/mnist', exist_ok=True)
        joblib.dump(mnist, os.path.join(p, 'mnist.pkl'))
    else:
        mnist = joblib.load(os.path.join(p, 'mnist.pkl'))
    return mnist


def preprocessing(mnist):
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)  # Convert from string to int

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, y_train, X_test, y_test


def show_img(instance):
    image = instance.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()


def train_SGD(X, y):
    model = SGDClassifier(random_state=42)
    model.fit(X, y)
    return model


def save_model(model, str):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', str))


def load_model(str):
    return joblib.load(os.path.join('models', str))


def accuracy_predictions(model, data, labels, plot=False):
    predictions = model.predict(data)
    n_correct = sum(predictions == labels)
    acc = 100*n_correct/len(labels)
    return acc


def cross_validation(model, X, y, n_splits):
    """
    Cross validation using stratified kfold
    That is, the data are splitted into X folds keeping the right ratio between the different classes!
    """
    skfolds = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    for train_index, test_index in skfolds.split(X, y):
        clone_model = clone(model)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_folds = X[test_index]
        y_test_folds = y[test_index]

        clone_model.fit(X_train_folds, y_train_folds)
        y_pred = clone_model.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))


def conf_matrix(model, X, y, n_splits, plot=False):

    y_train_pred = cross_val_predict(model, X, y, cv=n_splits)  # perform K-fold and return prediction on each test fold
    cm = confusion_matrix(y, y_train_pred)

    if plot:
        plt.imshow(cm, cmap='jet')
        plt.show()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    From such a plot, one can chose a suitable threshold for the application
    It depends what the user wants
    """
    plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.legend()
    plt.grid()
    plt.show()


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(precisions, recalls, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def prec_recall_threshold(model, X, y, n_splits):
    # Only for binary classification! -> change the y to True and False
    y = y == 5
    y_scores = cross_val_predict(model, X, y, cv=n_splits, method='decision_function')
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    if True:
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        plot_precision_vs_recall(precisions, recalls)

    # If we want to reach a specific precision -> select the precision
    precision_threshold = thresholds[np.argmax(precisions >= 0.9)]
    print(precision_threshold)
    # We predict:
    y_predict_90 = (y >= precision_threshold)
    # Check precision and recall (does not work...)
    if False:
        p = precision_score(y, y_predict_90)
        r = recall_score(y, y_predict_90)
        print(p, r)
        # In principle -> gives arond 90% precision and 40% recall
        # -> easy to have high precesion but then low recall


def prec_recall(model, X, y, n_splits):
    """
    Precision:
        evaluates the ratio of true positive to the total amount of positive (true + false positives)
        precision = TP/(TP+FP)
    Recall (/sensitivity/true positive rate):
        ratio of positive instances that are correctly detected by the classifier
        recall = TP/(TP+FN)
    F1 score:
        F1 favors classifiers with similar precision and recall
        combines precision and recall into a value that will have a low score if one of them is low
        F1 = 2/( (1/precision) + (1/recall) )
    """
    if False:
        y_train_pred = cross_val_predict(model, X, y, cv=n_splits)  # perform K-fold and return prediction on each test fold
        prec = precision_score(y, y_train_pred, average=None)
        recall = recall_score(y, y_train_pred, average=None)
        f1 = f1_score(y, y_train_pred, average=None)

    # precision-recall trade-off: increasing one decreases the other
    # Depending on where the threshold is to separate the class(es) -> high recall or high precision
    # scikit does not give access to the threshold but to the function returning the score
    # we can then manually set the threshold(s)
    y_scores = model.decision_function(X)
    if False:
        id = 7
        plt.plot(y.flatten(), y_scores[:, id].flatten(), '.', alpha=0.1)
        plt.show()

    # Which Threshold to use?
    prec_recall_threshold(model, X, y, n_splits)
    # print(y_scores, y_pred)

    # return prec, recall, f1


def plot_roc(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid()
    plt.show()


def roc(model, X, y, n_splits):
    """
    similar to precision/recall curve but
    plots true positive rate (=TPR=recall) against false positive rate (=FPR)
    TPR = TP/(TP+FN)            -> from the positive, how much are really positive
    FPR = FP/(FP+TN)            -> how much negative are miss-classified
    """
    y = y == 5
    y_scores = cross_val_predict(model, X, y, cv=n_splits, method='decision_function')
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    if False:
        plot_roc(fpr, tpr)
    # Another way to evaluate if good model is to lookg at area under ROC curve
    # perfect model AUC = 1, random model: AUC=0.5
    if False:
        auc_score = roc_auc_score(y, y_scores)
        print(auc_score)


def performance_measure_SGD(model, X, y):
    # Simple percentage of correctness in the training set
    acc = accuracy_predictions(model, X_train, y_train, plot=False)
    print(f'Accuracy {acc}%')

    # Cross-Validation
    if False:
        cross_validation(model, X, y, 3)
        # Accuracy is not a good measure:
        # model looking for not 5 in the set -> 90% because set everything to not 5
        # also bad for skewed dataset (different size of elements in classes)

    # Confusion-matrix
    if False:
        conf_matrix(model, X, y, 3, plot=True)

    # Precision and recall
    if True:
        prec_recall(model, X, y, 3)

    # ROC Curve: for binary classifiers
    if False:
        roc(model, X, y, 3)

    # To chose PR Curve or ROC Curve?
    # use PR if: few positive cases or do not care about false negative
    # else use ROC and ROC AUC


def performance_measure_RFC(model, X, y):
    # RFC does not have a decision_function but a predict_proba function
    y = y == 5
    y_probas = cross_val_predict(model, X, y, cv=3, method='predict_proba')
    y_scores = y_probas[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    if False:
        plot_roc(fpr, tpr)
    if False:
        auc_score = roc_auc_score(y, y_scores)
        print(auc_score)
        # -> much better results than SGD!
    if True:
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
        if True:
            plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            plot_precision_vs_recall(precisions, recalls)


if __name__ == '__main__':
    dl = False
    mnist = import_mnist(download=dl)

    X_train, y_train, X_test, y_test = preprocessing(mnist)
    # show_img(X_train[0])
    if False:
        if False:
            model = train_SGD(X_train, y_train)
            save_model(model, 'SGD.pkl')
        else:
            model = load_model('SGD.pkl')
    else:
        model = RandomForestClassifier(random_state=42)

    # Explore performance measurements
    if False:
        performance_measure_SGD(model, X_train, y_train)
    else:
        performance_measure_RFC(model, X_train, y_train)
