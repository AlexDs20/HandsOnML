#! python3
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def plot_digit(digit):
    image = digit.reshape(28, 28)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()


def show_img(model, X, y, id):
    prediction = model.predict([X[id]])
    label = y[id]
    image = X[id].reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.title(f'Label: {label}, Prediction: {prediction[0]}')
    plt.show()


def save_model(model, str):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', str))


def load_model(path_model, str):
    p = os.path.join(path_model, str)
    return joblib.load(p)


def preprocessing(mnist, scaling=False):
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)  # Convert from string to int

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Scaling
    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.astype(np.float64))

    return X_train, y_train, X_test, y_test


def import_mnist(path_data, download=False):
    p = os.path.join(path_data, 'mnist')
    if download:
        mnist = fetch_openml('mnist_784', version=1)
        os.makedirs(p, exist_ok=True)
        joblib.dump(mnist, os.path.join(p, 'mnist.pkl'))
    else:
        mnist = joblib.load(os.path.join(p, 'mnist.pkl'))
    return mnist


def check_score_prediction(model, digit):
    scores = model.decision_function(digit)
    print(f'All scores: {scores}')
    print(f'Highest score is in the {np.argmax(scores)} element of the scores')
    print(f'Looking into the classes_ the {np.argmax(scores)} corresponds to the predicted value:' +
          f'{model.classes_[np.argmax(scores)]}')


def select_model(path='./', train=False, save=False, strategy='', **kwargs):
    """
    train = False
    save = False
    strategy = 'OvO', 'OvR', ''

    **kwargs:
        model= 'SVC', 'SGD', 'KNN'
        X= data
        y= label
    """
    for key, value in kwargs.items():
        if key.lower() == 'model':
            model = value
        elif key.upper() == 'X':
            X = value
        elif key.lower() == 'y':
            y = value

    if model.upper() == 'SVC':
        mod = SVC()
    elif model.upper() == 'SGD':
        mod = SGDClassifier()
    elif model.upper() == 'KNN':
        mod = KNeighborsClassifier()

    if strategy.lower() == 'OvO'.lower():
        mod = OneVsOneClassifier(mod)
    elif strategy.lower() == 'OvR'.lower():
        mod = OneVsRestClassifier(mod)

    if train:
        mod.fit(X, y)
    if save:
        p = os.path.join(path,  model+'_'+strategy+'.pkl')
        save_model(mod, p)

    return mod


def error_analysis(model, X, y, n_splits):
    path_ex = os.path.join('/home/alexandre/Documents/Projects/HandsOnML','Exercises')
    if False:
        y_pred = cross_val_predict(model, X, y, cv=n_splits)
        joblib.dump(y_pred, os.path.join(path_ex, 'Chap3/ypred.pkl'))
    else:
        y_pred = joblib.load(os.path.join(path_ex, 'Chap3/ypred.pkl'))
    cm = confusion_matrix(y, y_pred)
    if False:
        plt.matshow(cm, cmap=plt.cm.gray)
        plt.show()
    if True:
        # Need to normalize by the number of each instance to properly normalize
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / row_sums
        # Put 0's on diagonal to only look at errors
        np.fill_diagonal(cm_norm, 0)
        plt.matshow(cm_norm, cmap=plt.cm.gray)
        plt.show()
        # -> many images get misclassified as 8!
        # But the row for 8 is not very bright -> meaning that most 8 get correctly classified


if __name__ == '__main__':
    """
    way to handle multiclass:
        *have many binary classification and compare their score. Associate the highest score
            -> Called one-versus-rest (OVR) / one-versus-all stategy
            -> prefered for most algorithm
        *many binary classifier: distinguish 0 and 1, 0 and 2, 1 and 2, ...
            -> Called one-versus-one (OvO) classifier (only needs to be trained on classed to be distinguised)
            -> good for classifiers that scales badly with size of set (e.g. SVM)
            -> N*(N-1)/2 classifier
            -> for an image, see which class wins the most duels

    Scikit-learn ses when multiclass, automatically runs OvR or OvO
    """
    # Paths
    path_ex = os.path.join('/home/alexandre/Documents/Projects/HandsOnML','Exercises')
    path_data = os.path.join(path_ex, 'data')
    path_model = os.path.join(path_ex, 'Chap3', 'models')

    dl = False
    mnist = import_mnist(path_data, download=dl)

    # Preprocessing
    X_train, y_train, X_test, y_test = preprocessing(mnist, scaling=False)

    # Choose model
    train = False
    save = False
    mod = 'SGD'
    strat = ''
    X = X_train[:-1]
    y = y_train[:-1]
    # model = select_model(train=True, save=True, model='SVC', strategy=''
    #                      X=X_train[:1000], y=y_train[:1000])

    if False:
        model = select_model(path_model, train=train, save=save, model=mod, strategy=strat, X=X, y=y)
        model = load_model(path_model, mod + '_' + strat + '.pkl')

    # if OVR -> estimators_ shows the 10 SVC, one for each category
    # print(model.estimators_)

    # Test:
    if False:
        show_img(model, X_train, y_train, 0)

    # check_score_prediction(model, [X_train[0]])

    # Evaluate the model using cross validation (kfolds)
    if False:
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        print(scores)
        # -> Scaling the data during preprocessing improves the accuracy

    # Error Analysis
    # In principle (automate):
    #       - Explore data preparation
    #       - try multiple-models (keep the best and get hyper-parameters: GridSearchCV)
    # error_analysis(model, X_train, y_train, 6)

    #--------------------------
    # MULTILABEL CLASSIFICATION
    #--------------------------
    if False:
        y_train_large = (y_train > 7)
        y_train_odd = (y_train % 2 == 1)
        y_multilabel = np.c_[y_train_large, y_train_odd]    # Concatenate

        model = select_model(path_model, train=False, save=False, model='KNN', X=X_train, y=y_multilabel)
        # show_img(model, X_train, y_train, 900)
        # To evaluate the model -> different possibilities
        # -> e.g. use the binary classifier metric and average over the labels
        # ex, for the F1 score:
        y_train_knn_pred = cross_val_predict(model, X_train, y_multilabel, cv=3)
        f1_score(y_multilabel, y_train_knn_pred, average="macro")

    #---------------------------
    # MULTIOUTPUT CLASSIFICATION
    #---------------------------
    # Generalisation of multilabel classification
    # -> each label can be multiclass -> more than 2 possible values
    # Here, denoising images -> multi-label (each pixel), multi-class (different int per pixel)
    if True:
        noise = np.random.randint(0, 100, (len(X_train), 28 * 28))
        X_train_mod = X_train + noise
        noise = np.random.randint(0, 100, (len(X_test), 28 * 28))
        X_test_mod = X_test + noise

        y_train_mod = X_train
        y_test_mod = X_test

        model = KNeighborsClassifier()
        model.fit(X_train_mod, y_train_mod)
        joblib.dump(model, 'KNN.pkl')
        clean_digit = model.predict([X_test_mod[0]])
        plot_digit(X_test_mod[0])
        plot_digit(clean_digit)
