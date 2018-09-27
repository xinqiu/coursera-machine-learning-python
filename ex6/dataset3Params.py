import numpy as np
import sklearn.svm as svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example,
    #                   predictions = svmPredict(model, Xval)
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using
    #        mean(double(predictions ~= yval))
    #
    C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    best_precision = 0.0

    for current_C in C_list:
        for current_sigma in sigma_list:
            model = svm.SVC(C=current_C, kernel='rbf', tol=1e-3, max_iter=200,
                            gamma=1.0 / (2.0 * current_sigma ** 2)).fit(X, y)
            predict_y = model.predict(Xval)
            precision = (predict_y == yval).astype(int).mean()

            if precision > best_precision:
                C, sigma = current_C, current_sigma
                best_precision = precision

    # =========================================================================
    return C, sigma
