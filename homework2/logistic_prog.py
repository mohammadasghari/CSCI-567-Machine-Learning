"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is logistic_bin_train, logistic_bin_test, logistic_mul_train, logistic_mul_test. You are also not supposed to modify any function other than these 4 functions. Please go through the instructions and comments in this file.
"""

import sys
import numpy
import json
import copy
import os

numpy.seterr(all='ignore')


def logistic_train_ovr(Xtrain, ytrain, w, b, step_size, max_iterations):
    '''
    Inputs:
    - Xtrain: training features, a N-by-D numpy array, where N is the number of training points and D is the dimensionality of features
    - ytrain: training labels, a N dimensional numpy array where N is the number of training points, indicating the labels of training data
    - w: D-dimensional vector, a numpy array which is the weight vector of logistic regression and initialized to be all 0s
    - b: scalar, which is the bias of logistic regression and initialized to be 0
    - step_size: step size (learning rate)
    - max_iterations: the maximum number of iterations

    Returns:
    - learnt w_l and b_l

    To help you verify your results, we have provided a function named logistic_train_bin in this file, it serves for the purpose of training a binary logistic regression model.
    '''

    # you need to fill in your solution here

    Xtrain_tilde = numpy.hstack([numpy.ones([Xtrain.shape[0], 1]), Xtrain])

    w_tilde = numpy.hstack([b, w])

    for iter in range(0, max_iterations):

        gradient_term = [0] * len(w_tilde)

        for n in range(0, Xtrain_tilde.shape[0]):
            expo_term = numpy.exp(-numpy.matmul(w_tilde, Xtrain_tilde[n, :]))

            pred_prob = 1 / (1 + expo_term)

            gradient_term += (pred_prob - ytrain[n]) * Xtrain_tilde[n, :]

        w_tilde = w_tilde - step_size * gradient_term / Xtrain_tilde.shape[0]

    b_l = w_tilde[0]
    w_l = w_tilde[1:]

    return w_l, b_l


def logistic_test_ovr(Xtest, w_l, b_l):
    """
    Inputs:
    - Xtest: testing features, a N-by-D numpy array, where N is the number of test points and D is the dimensionality of features.
    - ytest: testing labels, a N dimensional numpy array where N is the number of test points, indicating the labels of test data.
    - w_l: a numpy array of D elements as a D-dimensional vector, which is the weight vector of logistic regression and learned by logistic_train().
    - b_l: a scalar, which is the bias of logistic regression and learned by logistic_train().

    Returns:
    - a numpy array with num_test elements predicted probability , i.e. the output of your logistic function, of each testing data.
    """
    # you need to fill in your solution here

    test_pred = []

    for n in range(0, Xtest.shape[0]):
        expo_term = numpy.exp(-numpy.matmul(w_l, Xtest[n, :]) - b_l)

        pred_output = 1 / (1 + expo_term)

        test_pred.append(pred_output)

    return test_pred


def logistic_mul_train(Xtrain, ytrain, w, b, step_size, max_iterations):
    '''
    Inputs:
    - Xtrain: training features, a N-by-D numpy array, where N is the number of training points and D is the dimensionality of features.
    - ytrain: training labels, a N-by-C numpy array, where N is the number of training points and C is the number of classes.
      Each row of ytrain is a C-dimensional one-hot vector with the true class as 1 and otherwise 0
    - w: the weight vector of multinomial logistic regression which is C-by-D dimension and initialized to be all 0s
    - b: the bias of logistic multinomial regression which a vector with size C and initialized to be 0
    - step_size: step size (learning rate)
    - max_iterations: the maximum number of iterations

    Returns:
    - learnt w and b, denoted as w_l and b_l.
    '''

    # you need to fill in your solution here

    Xtrain_tilde = numpy.hstack([numpy.ones([Xtrain.shape[0], 1]), Xtrain])

    w_tilde = numpy.vstack([b, w.transpose()]).transpose()

    for iter in range(0, max_iterations):

        gradient = numpy.zeros(w_tilde.shape)

        for n in range(0, Xtrain_tilde.shape[0]):
            x_n_tilde = Xtrain_tilde[n, :]


            SF_num =[]
            SF_input = []

            for l in range(0, w_tilde.shape[0]):

                SF_input.append(numpy.matmul(w_tilde[l, :], x_n_tilde))

            SF_input_max = max(SF_input)

            SF_input_tilde = [SF_input[i] - SF_input_max for i in range(0,len(SF_input))]

            for l in range(0, w_tilde.shape[0]):

                SF_num.append(numpy.exp(SF_input_tilde[l]))

            SF = SF_num/sum(SF_num)

            for k in range(0, len(SF)):

                gradient[k, :] += (SF[k] - ytrain[n, k]) * x_n_tilde

        for k in range(0, w_tilde.shape[0]):
            w_tilde[k, :] = w_tilde[k, :] - step_size * gradient[k, :] / Xtrain_tilde.shape[0]

    b_l = w_tilde[:, 0]
    w_l = w_tilde[:, 1:]

    return w_l, b_l


def logistic_mul_test(Xtest, w_l, b_l):
    """
    Inputs:
    - Xtest: testing features, a N-by-D numpy array where N is the number of test points and D is the dimensionality of features.
    - ytest: testing labels, a N-by-C numpy array, where N is the number of test points and C is the number of classes.
      Each row of ytest is a C-dimensional one-hot vector with the true class as 1 and otherwise 0.
    - w_l: the weight vector of multinomial logistic regression which is C-by-D dimension, learned by mul_logistic_train().
    - b_l: the bias of logistic multinomial regression which a vector with size C, learned by mul_logistic_train().

    Returns:
    - test_pred: a N-by-C numpy array predicting 10-class probability of each testing data, i.e. the output of your logistic function.
    """
    # you need to fill in your solution here

    test_pred = numpy.zeros((Xtest.shape[0], len(b_l)))

    for n in range(Xtest.shape[0]):

        x_n = Xtest[n, :]

        test_all_classes = []
        for k in range(len(b_l)):
            test_all_classes.append(numpy.matmul(x_n, w_l[k, :]) + b_l[k])

        class_index_maximizer = numpy.argmax(test_all_classes)

        test_pred[n, class_index_maximizer] = 1

    return test_pred


###########################################################################
#          Please DO NOT change the following parts of the script         #
###########################################################################

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def logistic_train_bin(Xtrain, ytrain, w, b, step_size, max_iterations):
    '''
    Go through logistic_train_ovr() before you get into this function.

    Inputs:
    - Xtrain: training features, a N-by-D numpy array, where N is the number of training points and D is the dimensionality of features
    - ytrain: training labels, a N dimensional numpy array where N is the number of training points, indicating the labels of training data
    - w: D-dimensional vector, a numpy array which is the weight vector of logistic regression and initialized to be all 0s
    - b: scalar, which is the bias of logistic regression and initialized to be 0
    - step_size: step size (learning rate)
    - max_iterations: the maximum number of iterations

    Returns:
    - learnt w and b
    '''

    m, n = Xtrain.shape
    for i in range(0, max_iterations):
        pred = sigmoid(numpy.dot(Xtrain, w) + b)
        b += (-1) * step_size * (1.0 / m) * sum(pred - ytrain)
        w += numpy.array([(-1) * step_size * (1.0 / m) * (sum((pred - ytrain) * Xtrain[:, j])) for j in range(n)])

    return w, b


def data_loader_mnist(dataset, num_cls=10):
    '''
    Xtrain: the training feature of mnist dataset, numpy array with num_train (# of training data) by D (feature dimension: 784)
    ytrain: the training label of mnist dataset in the forms of 0~9, numpy array with num_train (# of training data)
    ytrain_mat: the training label of mnist dataset in the forms of length-10 one-hot vector, e.g. data with label '0' will be represented as a vector v which is labeled as 1 on v[0] and 0 from v[1] to v[9], numpy array with num_train (# of training data), numpy array with N_tr (# of training data) by 10 (# of categories)
    Xvalid, yvalid, yvalid_mat, Xtest, ytest, ytest_mat are defined in a similar way for validation and testing set, respectively.
    '''

    with open(dataset, 'r') as f:
        data_set = json.load(f)

    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = numpy.array(train_set[0])
    ytrain = numpy.array(train_set[1])
    ytrain_mat = numpy.zeros((len(ytrain), num_cls))
    for i in range(len(ytrain)):
        ytrain_mat[i, ytrain[i]] = 1

    Xvalid = numpy.array(valid_set[0])
    yvalid = numpy.array(valid_set[1])
    yvalid_mat = numpy.zeros((len(yvalid), num_cls))
    for i in range(len(yvalid)):
        yvalid_mat[i, yvalid[i]] = 1

    Xtest = numpy.array(test_set[0])
    ytest = numpy.array(test_set[1])
    ytest_mat = numpy.zeros((len(ytest), num_cls))
    for i in range(len(ytest)):
        ytest_mat[i, ytest[i]] = 1

    return Xtrain, ytrain, ytrain_mat, Xvalid, yvalid, yvalid_mat, Xtest, ytest, ytest_mat


def mnist_convert_bin(Xfea_tr, Xlabel_tr, i):
    '''
    re-label the 10 categorical data to binary, given the input i indicating which is the positive class.
    To balance the training data, we take the same number of negative samples as positive samples.
    '''

    Xlabel_tr_bin = copy.copy(Xlabel_tr)
    label_pos = numpy.where(Xlabel_tr == i)
    label_neg = numpy.where(Xlabel_tr != i)

    mask = numpy.ones(len(Xlabel_tr), dtype=bool)
    mask[label_neg[0][len(label_pos[0]):]] = False
    Xfea_tr_bin = Xfea_tr[mask]

    Xlabel_tr_bin[label_pos] = 1
    Xlabel_tr_bin[label_neg] = 0
    Xlabel_tr_bin = Xlabel_tr_bin[mask]

    return Xfea_tr_bin, Xlabel_tr_bin


def eval_multi_class(test_bin_prob, ytest):
    test_mul_pred = numpy.zeros(ytest.shape[0])
    for i in range(ytest.shape[0]):
        test_mul_pred[i] = numpy.argmax(test_bin_prob[i, :])
    test_acc = float(sum(test_mul_pred == ytest)) / len(ytest)
    return test_acc


def ovr_logistic(Xtrain, ytrain, ytrain_mat, Xtest, ytest, ytest_mat, num_cls=10):
    m, n = numpy.array(Xtrain).shape
    test_bin_prob = numpy.zeros((ytest.shape[0], num_cls))

    for i in range(num_cls):
        w = numpy.zeros(n)
        b = 0
        step_size = 0.5
        max_iterations = 200

        Xtrain_bin, ytrain_bin = mnist_convert_bin(Xtrain, ytrain, i)

        w_l, b_l = logistic_train_ovr(Xtrain_bin, ytrain_bin, w, b, step_size, max_iterations)
        test_bin_prob[:, i] = logistic_test_ovr(Xtest, w_l, b_l)

    test_acc = eval_multi_class(test_bin_prob, ytest)
    return test_acc


def multinomial_logistic(Xtrain, ytrain, ytrain_mat, Xtest, ytest, ytest_mat, num_cls=10):
    # training data
    m, n = numpy.array(Xtrain).shape
    w = numpy.zeros((num_cls, n))
    b = numpy.zeros(num_cls)
    step_size = 0.01
    max_iterations = 200

    w_l, b_l = logistic_mul_train(Xtrain, ytrain_mat, w, b, step_size, max_iterations)
    test_mul_prob = logistic_mul_test(Xtest, w_l, b_l)

    test_acc = eval_multi_class(test_mul_prob, ytest)
    return test_acc


if __name__ == "__main__":
    num_cls = 10
    test_acc = dict()
    '''
    data loading
    '''
    Xtrain, ytrain, ytrain_mat, _, _, _, Xtest, ytest, ytest_mat = data_loader_mnist(dataset='mnist_subset.json',
                                                                                     num_cls=num_cls)
    '''
    multi-class classification
    '''

    test_bin_acc = ovr_logistic(Xtrain, ytrain, ytrain_mat, Xtest, ytest, ytest_mat, num_cls)
    test_mul_acc = multinomial_logistic(Xtrain, ytrain, ytrain_mat, Xtest, ytest, ytest_mat, num_cls)

    print('test_bin_acc: %.4f, test_mul_acc:%.4f' % (test_bin_acc, test_mul_acc))
    '''
    save the results
    '''
    test_acc['bin'] = test_bin_acc
    test_acc['mul'] = test_mul_acc
    with open('logistic_res.json', 'w') as f_json:
        json.dump(test_acc, f_json)