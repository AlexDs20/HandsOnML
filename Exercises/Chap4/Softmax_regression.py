# Softmax Regression
#
# -> generalization of logistic regression to support several classes
#    without having to combine multiple binary classifiers
#
#   Idea:
#       for an instance x -> computes a score s_k(x) for each class k
#       the estimate the probability of each class by applying softmax function
#
#       s_k(x) = x^t \theta^(k)    -> each class has its own set of \theta
#
#       Often all the \theta from all classes are saved into rows in a parameter matrix  \theta
#
#       Once score is calculated for each class -> calculate the probability that
#       the instance belongs to class k.
#
#       prob: exponential of score / normalization
#           p = \sigma(s(x))_k = exp(s_k(x))/\sum_{j=1}^K exp(s_j(x))
#
#       -> Estimates class with the highest probability:
#           \hat{y} = argmax_k s_k(x)
#
#   Softmax predicts only one class at a time! (it is multi-class, not multi-output)
#
#   Training:
#       Want to have high prob for the target class and low prob for the others
#       -> minimiz the cost function (cross-entropy)
#
#   Cost function (cross-entropy):
#       J(\theta) = -(1/m) \sum_i=1^m \sum_k=1^K y_k^(i) \log(\hat{p}_k^(i))
#
#       y_k^(i) = target probability (usually 0 or 1 = is class or is not)
#

# Use softmax to classify iris flows in the three classes
if True:
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    import numpy as np
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = iris["target"]

    # By default logisticregression uses one-versus the rest when more than 2 classes
    # but -> multi_class="multinomial" -> to switch to softmax
    # must specify solver that suppors softmax
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    model.fit(X, y)
    #               p1, p2    parameters
    model.predict([[5, 2]])
    model.predict_proba([[5, 2]])
