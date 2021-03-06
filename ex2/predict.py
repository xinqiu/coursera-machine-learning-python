import numpy as np
from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#


# =========================================================================
    p = sigmoid(X.dot(theta.T))
    vfunc = np.vectorize(lambda y: 1 if y >= 0.5 else 0)
    return vfunc(p)