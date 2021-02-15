# EECS 445 - Winter 2021
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string
import nltk

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer

from helper import *

def clean_string(text):
    for element in text:
        text = text.replace(element, element.lower())
        if element in (string.punctuation):
            text = text.replace(element, " ")
    return text.split()

def extract_dictionary(df):
    """
    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    all_words = []
    for index, row in df.iterrows():
        text = row['reviewText']
        for element in text:
            text = text.replace(element, element.lower())
            if element in (string.punctuation):
                text = text.replace(element, " ")
        split_words = text.split()
        all_words.extend(split_words)

    idx = 0
    for word in all_words:
        if word not in word_dict:
            word_dict[word] = idx
            idx += 1

    return word_dict

# NOTE: This function for lematization is copied from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (# of reviews, # of words in dictionary).
    Input:
        df: dataframe that has the ratings and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    idx = 0
    for index, row in df.iterrows():
        empty_list = []
        text = row['reviewText']
        cleaned_list = set(clean_string(text))
        for word in word_dict:
            if word in cleaned_list:
                empty_list.append(1)
            else:
                empty_list.append(0)
        a = np.array(empty_list)
        feature_matrix[idx] = a
        idx += 1

    return feature_matrix

def generate_feature_matrix_challenge(corpus_train, corpus_test):
    """
    Reads a dataframe and the corpus of all lematized words
    to generate a tf-idf matrix 
    Input:
        df: dataframe that has the ratings and labels
        corpus: corpus of all reviews
    Returns:
        a feature matrix of dimension (# of reviews, # of distinct words in corpus)
    """
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(corpus_train)
    X_test = vectorizer.transform(corpus_test)

    return (X_train, X_test)

def extract_dictionary_challenge(df):
    """
    Reads a pandas dataframe, and returns a corpus consisting of each review with its lemmatized words
    Input:
        df: dataframe/output of load_data()
    Returns:
        a corpus consisting of each review with its lemmatized words
    """
    word_dict = {}
    corpus = []
    lemmatizer = WordNetLemmatizer()
    for index, row in df.iterrows():
        text = row['reviewText']
        for element in text:
            text = text.replace(element, element.lower())
            if element in (string.punctuation):
                text = text.replace(element, " ")

        # NOTE: This line of code for lemmatization is copied from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
        lemmatized_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
        final_string = " ".join(lemmatized_text)
        corpus.append(final_string)
    return corpus

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn = confusion_matrix[0][0]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1]

    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred, zero_division=0)
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred, zero_division=0)
    elif metric == "specificity":
        if (tn + fp) == 0:
            return 0
        else:
            return tn/(tn + fp)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_pred, zero_division=0)
    
    print("Invalid metric")
    assert(False)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful
    #SKF
    skf = StratifiedKFold(n_splits=k)

    #Put the performance of the model on each fold in the scores array
    scores = []
    for train_index, test_index in skf.split(X, y):
        # Train our model
        training_data_x = X[train_index, :]
        training_data_y = y[train_index]
        clf.fit(training_data_x, training_data_y)

        # Test the model
        if metric == "auroc":
            y_pred = clf.decision_function(X[test_index])
            y_true = y[test_index]
        else:
            y_pred = clf.predict(X[test_index])
            y_true = y[test_index]
        scores.append(performance(y_true, y_pred, metric=metric))
    
    # And return the average performance across all fold splits.
    return np.array(scores).mean()

def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    if penalty == 'l1':
        return LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced')
    if degree > 1:
        return SVC(kernel='poly', degree=degree, C=c, coef0=r, class_weight=class_weight, gamma='auto')
    # For question 2c
    x = SVC(kernel='linear', C=c, class_weight=class_weight)
    return x

def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    C_results = {}
    for c_value in C_range:
        clf = select_classifier(penalty=penalty, c=c_value)
        C_results[c_value] = cv_performance(clf, X, y, k, metric)
    sorted_c = {k: v for k, v in sorted(C_results.items(), key=lambda item: (item[1], -item[0]), reverse=True)}
    l = list(sorted_c.items())
    best_C_val=l[0][0]
    print("Performance measure: ", metric, " | Best C value of: ", best_C_val, " | Performance: ", l[0][1])
    return best_C_val


def plot_weight(X,y,penalty,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = select_classifier(c=c, penalty=penalty)
        clf.fit(X, y)
        theta = clf.coef_[0]
        
        l0 = np.count_nonzero(theta)
        norm0.append(l0)
    


    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()

def print_bar(X,y, words):
    clf = select_classifier(c=0.1)
    clf.fit(X,y)

    theta = clf.coef_[0]
    theta_arg_sorted = np.argsort(theta)
    largest_indices = theta_arg_sorted[-10:]
    smallest_indices = theta_arg_sorted[:10]
    keys_list = np.array(list(words))

    largest_list = keys_list[largest_indices]
    smallest_list = keys_list[smallest_indices]

    y_large = theta[largest_indices]
    y_small = theta[smallest_indices]

    plt.figure(figsize=(9,6))
    plt.bar(largest_list, y_large)
    plt.xlabel('Word')
    plt.ylabel('Coefficient')
    plt.title('Words vs. Theta Coefficient: 10 most positive words')
    plt.savefig('Most_positive.png')
    plt.close()

    plt.figure(figsize=(9,6))
    plt.bar(smallest_list, y_small)
    plt.xlabel('Word')
    plt.ylabel('Coefficient')
    plt.title('Words vs. Theta Coefficient: 10 most negative words')
    plt.savefig('Most_negative.png')
    plt.close()
def sign(value):
    if value > 1:
        return 1
    elif value < 1:
        return -1
    else:
        assert(False)
def predict_perceptron(X_test, theta, b):
    dot = np.dot(X_test, theta)
    x = np.array([b])
    final = dot + x
    vectorized_func = np.vectorize(sign)
    y_pred = vectorized_func(final)
    return y_pred


def train_perceptron(X_train, Y_train):
    """
    Takes in an input training data X and labels y and 
    returns a valid decision boundary theta, b found through
    the Perceptron algorithm. If a valid decision boundary 
    can't be found, this function fails to terminate.

    # NOTE: if you use the first 500 points of the dataset 
    # we have provided, this functions should converge
    """

    k = 0
    theta = np.zeros(X_train.shape[1])
    b = 0
    mclf = True
    while mclf:
        mclf = False
        for i in range(len(X_train)):
            if Y_train[i] * (np.dot(theta, X_train[i]) + b) <= 0:
                theta = theta + 0.1 * (Y_train[i] -  np.dot(theta, X_train[i])) * X_train[i]
                b += Y_train[i]
                mclf = True
                k += 1
    return theta, b



def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            param_range: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter values for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance as a pair (C,r)
    """
    best_C_val,best_r_val = 0.0, 0.0
    C_results = {}
    for i in range(len(param_range)):
        c_value = param_range[i][0]
        r_value = param_range[i][1]
        clf = select_classifier(penalty='l2', c=c_value, degree=2, r=r_value, class_weight='balanced')
        C_results[(c_value, r_value)] = cv_performance(clf, X, y, k, metric)
    sorted_c = {k: v for k, v in sorted(C_results.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]), reverse=True)}
    l = list(sorted_c.items())
    best_C_val = l[0][0][0]
    best_r_val = l[0][0][1]

    print("Best C: ", best_C_val, " | Best r: ", best_r_val, " | Metric ", metric, ": ", l[0][1])
    print(sorted_c)
    return best_C_val,best_r_val

def performance_challenge(y_true, y_pred, ignore):
    """
    Calculates the accuracy metric as evaluated on the true labels
    y_true versus the predicted labels y_pred. Ignores any results with label ignore
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    y_true_ignore = y_true[np.argwhere(y_true != ignore)]
    y_pred_ignore = p_pred[np.argwhere(y_true != ignore)]
    confusion_matrix = metrics.confusion_matrix(y_true_ignore, y_pred_ignore)
    
    return metrics.accuracy_score(y_true, y_pred)

def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    # IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # TODO: Questions 2, 3, 4
    # Question 1c
    # t = get_split_binary_data()
    # unique_words = (t[0].shape)[1]
    # print("Number of unique words: ", unique_words)

    # words_per_review = np.sum(t[0], axis=1)
    # average = np.average(words_per_review)
    # print("Average number of non-zero features per rating: ", average)

    # word_count = np.sum(t[0], axis=0)
    # print(t[0][800])
    # index_max = np.argmax(word_count)
    # word_at_index = list(t[4].keys())[list(t[4].values()).index(index_max)]
    # print("Word appearing in the most number of reviews: ", word_at_index)

    # Question 3D
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # C_range = np.array([0.001, 0.01, 0.1, 1., 10., 100., 1000.])
    # select_param_linear(X_train, Y_train, 5, 'accuracy', C_range, 'l2')
    # select_param_linear(X_train, Y_train, 5, 'f1-score', C_range, 'l2')
    # select_param_linear(X_train, Y_train, 5, 'auroc', C_range, 'l2')
    # select_param_linear(X_train, Y_train, 5, 'precision', C_range, 'l2')
    # select_param_linear(X_train, Y_train, 5, 'sensitivity', C_range, 'l2')
    # select_param_linear(X_train, Y_train, 5, 'specificity', C_range, 'l2')

    # Question 2E
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # C_range = np.array([0.1])
    # clf = select_classifier(c=0.1)
    # clf.fit(X_train, Y_train)
    # y_pred = clf.predict(X_test)
    # acc = performance(Y_test, y_pred, "accuracy")
    # f1 = performance(Y_test, y_pred, "f1-score")
    # prec = performance(Y_test, y_pred, "precision")
    # sens = performance(Y_test, y_pred, "sensitivity")
    # spec = performance(Y_test, y_pred, "specificity")
    # y_pred = clf.decision_function(X_test)
    # auroc = performance(Y_test, y_pred, "auroc")
    # print("Accuracy: ", acc)
    # print("F1-score: ", f1)
    # print("Auroc: ", auroc)
    # print("Precision: ", prec)
    # print("Sensitivity: ", sens)
    # print("Specificity: ", spec)

    # Question 3F
    # print("============================QUESTION 3.1 F/G============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # C_range = np.array([0.001, 0.01, 0.1, 1., 10., 100., 1000.])
    # plot_weight(X_train, Y_train, penalty="l2", C_range=C_range)

    # print("============================QUESTION 3.1 H============================")
    # print_bar(X_train, Y_train, dictionary_binary)

    # print("============================QUESTION 3.2 B============================")
    # print("GRID SEARCH")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(class_size=50)
    # param_range = [[C, r] for C in np.logspace(-3, 3, 7) for r in np.logspace(-3, 3, 7)]
    # select_param_quadratic(X_train, Y_train, 5, metric="auroc", param_range=param_range)
    # powers = np.random.uniform(-3.0, 3.0, (25, 2))
    # tens = np.full((25,2), 10)
    # new_param_range = tens ** powers
    # select_param_quadratic(X_train, Y_train, 5, metric="auroc", param_range=new_param_range)

    # Question 3.4 a
    # print("============================QUESTION 3.4 A ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # C_range = np.array([0.001, 0.01, 0.1, 1.])
    # select_param_linear(X_train, Y_train, 5, 'auroc', C_range, 'l1')

    # print("============================QUESTION 3.4 B ============================")
    # C_range = np.array([0.001, 0.01, 0.1, 1.])
    # plot_weight(X_train, Y_train, 'l1', C_range)

    # print("============================QUESTION 3.5 A ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # theta, b = train_perceptron(X_train, Y_train)
    # y_pred = predict_perceptron(X_test, theta, b)
    # acc = performance(Y_test, y_pred, "accuracy")
    # print("Accuracy: ", acc)

    # print("============================QUESTION 4.1 B/C ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # class_weight = {-1: 1, 1: 10}
    # clf = select_classifier(c=0.1, class_weight=class_weight)
    # clf.fit(X_train, Y_train)
    # y_pred = clf.predict(X_test)
    # acc = performance(Y_test, y_pred, "accuracy")
    # f1 = performance(Y_test, y_pred, "f1-score")
    # prec = performance(Y_test, y_pred, "precision")
    # sens = performance(Y_test, y_pred, "sensitivity")
    # spec = performance(Y_test, y_pred, "specificity")
    # y_pred = clf.decision_function(X_test)
    # auroc = performance(Y_test, y_pred, "auroc")
    # print("Accuracy: ", acc)
    # print("F1-score: ", f1)
    # print("Auroc: ", auroc)
    # print("Precision: ", prec)
    # print("Sensitivity: ", sens)
    # print("Specificity: ", spec)

    # print("============================QUESTION 4.2 A/B ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    # IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # class_weight = {-1: 1, 1: 1}
    # clf = select_classifier(c=0.1, class_weight=class_weight)
    # clf.fit(IMB_features, IMB_labels)
    # y_pred = clf.predict(IMB_test_features)
    # acc = performance(IMB_test_labels, y_pred, "accuracy")
    # f1 = performance(IMB_test_labels, y_pred, "f1-score")
    # prec = performance(IMB_test_labels, y_pred, "precision")
    # sens = performance(IMB_test_labels, y_pred, "sensitivity")
    # spec = performance(IMB_test_labels, y_pred, "specificity")
    # y_pred = clf.decision_function(IMB_test_features)
    # auroc = performance(IMB_test_labels, y_pred, "auroc")
    # print("Accuracy: ", acc)
    # print("F1-score: ", f1)
    # print("Auroc: ", auroc)
    # print("Precision: ", prec)
    # print("Sensitivity: ", sens)
    # print("Specificity: ", spec)

    # print("============================QUESTION 4.3 A ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    # IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)
    # weight_class_range = [w1 for w1 in np.linspace(0,1,100,endpoint=False)]
    # max_performance = 0
    # results = {}
    # for w1_value in weight_class_range:
    #     if(w1_value != 0):
    #         class_weight = {-1: w1_value, 1: 1 - w1_value}
    #         clf = select_classifier(c=0.1, class_weight=class_weight)
    #         results[w1_value] = cv_performance(clf, IMB_features, IMB_labels, 5, "f1-score")
    # a = {k: v for k, v in sorted(results.items(), key=lambda item: (item[1], -item[0]), reverse=True)}
    # l = list(a.items())
    # best_w1_val=l[0][0]
    # print("Performance measure: auroc ", "| Best w1 value of: ", best_w1_val, " | Performance: ", l[0][1])
    # # print(a)

    # print("============================QUESTION 4.3 C ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    # IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # class_weight = {-1: 0.54, 1: 0.46}
    # clf = select_classifier(c=0.1, class_weight=class_weight)
    # clf.fit(IMB_features, IMB_labels)
    # y_pred = clf.predict(IMB_test_features)
    # acc = performance(IMB_test_labels, y_pred, "accuracy")
    # f1 = performance(IMB_test_labels, y_pred, "f1-score")
    # prec = performance(IMB_test_labels, y_pred, "precision")
    # sens = performance(IMB_test_labels, y_pred, "sensitivity")
    # spec = performance(IMB_test_labels, y_pred, "specificity")
    # y_pred = clf.decision_function(IMB_test_features)
    # auroc = performance(IMB_test_labels, y_pred, "auroc")
    # print("Accuracy: ", acc)
    # print("F1-score: ", f1)
    # print("Auroc: ", auroc)
    # print("Precision: ", prec)
    # print("Sensitivity: ", sens)
    # print("Specificity: ", spec)

    # print("============================QUESTION 4.4 ============================")
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    # IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    # IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # class_weight = {-1: 0.54, 1: 0.46}
    # clf = select_classifier(c=0.1, class_weight=class_weight)

    # plt.figure(0).clf()
    # clf.fit(IMB_features, IMB_labels)
    # pred = clf.decision_function(IMB_test_features)
    # fpr, tpr, thresh = metrics.roc_curve(IMB_test_labels, pred)
    # auc = metrics.roc_auc_score(IMB_test_labels, pred)
    # plt.plot(fpr,tpr,label="F1-score optimized weights "+str(auc))

    # clf2 = select_classifier(c=0.1, class_weight={-1:1, 1:1})
    # clf2.fit(IMB_features, IMB_labels)
    # pred2 = clf2.decision_function(IMB_test_features)
    # fpr, tpr, thresh = metrics.roc_curve(IMB_test_labels, pred2)
    # auc = metrics.roc_auc_score(IMB_test_labels, pred2)
    # plt.plot(fpr,tpr,label="Balanced weights "+str(auc))

    # plt.legend(loc=0)
    # plt.title('ROC Curve')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("False Negative Rate")


    # plt.savefig('Auc_figure.png')
    # plt.close()
    

    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    # X_train, Y_train, X_test = get_multiclass_training_data(class_size=50, get_test=False)
    # select_param_linear(X_train, Y_train, 5, 'accuracy', C_range, 'l2')
    # print("hello")
    y_true = [0, 1, -1, 0, 1, 1]
    y_pred = [1, 1, -1, -1, 1, 1]
    performance_challenge(y_true, y_pred, 0)



    

if __name__ == '__main__':
    main()
