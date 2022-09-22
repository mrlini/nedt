"""
Naive Equilibrium Decision Tree (NEDT) 
- decision tree based on Nash equilibria for multi-class classification

requirements: numba, scipy, math, numpy
"""
from math import fabs, erf, erfc
import numpy as np

from numba import njit, float64, vectorize
from scipy.stats import norm


NPY_SQRT1_2 = 1.0 / np.sqrt(2)


@njit(float64(float64), cache=True, fastmath=True)
def ndtr_numba(a):
    """Gaussian cumulative distribution function using numba"""

    if (np.isnan(a)):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if (z < NPY_SQRT1_2):
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)
        if (x > 0):
            y = 1.0 - y

    return y


@vectorize([float64(float64)], cache=True, fastmath=True)
def ndtrVec_numba(x):
    return ndtr_numba(x)


@njit("f8[:](f8[:, :], f8[:])", fastmath=True, cache=True)
def np_dot_numba(x, beta):
    """
    implement np.dot() using numba

    Args:
        X (np.ndarray): instances
        beta (np.ndarray): beta vector

    Returns:
        np.ndarray(): vector that contains the equivalent of np.dot(X, beta)
    """
    rez = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        _dot = 0
        for j in range(x.shape[1]):
            _dot += x[i][j]*beta[j]
        rez[i] = _dot
    return rez


@njit("f8(i8[:])", fastmath=True, cache=True)
def entropy_numba(y):
    """
    entropy

    Args:
        y (np.ndarray): vector of labels

    Returns:
        double: entropy
    """
    class_labels = np.unique(y)
    entropy = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        entropy += -p_cls * np.log2(p_cls)
    return entropy


@njit("f8(i8[:])", fastmath=True, cache=True)
def gini_numba(y):
    """
    gini

    Args:
        y (np.ndarray): vector of labels

    Returns:
        double: gini index
    """
    classes = np.unique(y)
    m = y.size
    sum = 0
    for c in classes:
        sum += (np.sum(y == c)/m)**2
    return 1-sum


@njit("f8(f8[:, :], i8[:], f8[:], i8)", fastmath=True, cache=True)
def payoff_nod_numba(X, y, Beta, nod):
    """
    entropy for node, beta_left and beta_right

    Args:
        X (np.ndarray()): data instances
        y (np.ndarray()): true labels
        Beta (np.ndarray()): beta_left and beta_right as a vector (beta to classify 0 (beta_left) and beta to classify 1 (beta_right))
        nod (int): for which node to compute the payoff 0:'stanga' 1:'dreapta'

    Returns:
        double: entropy for node
    """
    marime_beta = X.shape[1]
    beta_left = Beta[:marime_beta]
    beta_right = Beta[marime_beta:]
    beta_nou = np.zeros(marime_beta)

    for i in range(marime_beta):
        beta_nou[i] = (beta_left[i] + beta_right[i]) / 2.0

    produs_nou = np_dot_numba(X, beta_nou)
    if nod == 0:
        y_nou = y[produs_nou < 0]
    elif nod == 1:
        y_nou = y[produs_nou >= 0]
    else:
        print('node should be 0:left or 1:right')
    return entropy_numba(y_nou)


@njit(fastmath=True, cache=True)
def payoff_toti_numba(X, y, Beta):
    """
    payoff for the two nodes
    """
    marime_beta = X.shape[1]
    beta_left = Beta[:marime_beta]
    beta_right = Beta[marime_beta:]
    beta_nou = np.zeros(marime_beta)

    for i in range(marime_beta):
        beta_nou[i] = (beta_left[i] + beta_right[i]) / 2.0

    produs_nou = np_dot_numba(X, beta_nou)
    y_nou_stanga = y[produs_nou < 0]
    y_nou_dreapta = y[produs_nou >= 0]

    return entropy_numba(y_nou_stanga), \
        entropy_numba(y_nou_dreapta)


@njit("f8(f8[:], f8[:, :], i8[:], i8, f8, f8, b1)", fastmath=True, cache=True)
def devieri_jucatori_one_for_numba(Beta, X, y, nr_devieri, loc_random, scale_random, zgomot_normal):
    """
    Beta must be a vector

    Args:
        Beta (np.ndarray()): strategy vector (beta_left and beta_right)
        X (np.ndarray()): instances
        y (np.ndarray()): labels
        nr_devieri (int): no. deviations to compute
        loc_random (double): zgomot_normal=True - mu for random noise, zgomot_normal=False - low limit for uniform noise
        scale_random (double): zgomot_normal=True - sigma for random noise, zgomot_normal=False - upper limit for uniform noise
        zgomot_normal (bool, optional): tip zgomot, True - normal, False - uniform. Defaults to False.

    Returns:
        [type]: [description]
    """
    marime_beta = X.shape[1]
    payoffs = payoff_toti_numba(X, y, Beta)
    v = 0

    for jucator in range(2):
        payoff_nou = np.zeros(nr_devieri)
        for i in range(nr_devieri):
            _beta = Beta.copy()
            for j in range(marime_beta):
                if zgomot_normal:
                    _beta[jucator * marime_beta + j] = Beta[
                        jucator * marime_beta + j] + np.random.normal(
                        loc=loc_random, scale=scale_random)
                else:
                    _beta[jucator * marime_beta + j] = Beta[
                        jucator * marime_beta + j] + np.random.uniform(loc_random, scale_random)
            payoff_nou[i] = payoff_nod_numba(X, y, _beta, jucator)
        diferente = payoffs[jucator] - payoff_nou
        v += np.sum(diferente[diferente > 0] ** 2)
    return v


def check_purity(y):
    """
    check if y contains only one label
    """
    clase = np.unique(y)
    if len(clase) == 1:
        return True
    else:
        return False


def create_leaf(X, y, no_classes):
    """
    check from which class the majority of samples are present

    Returns:
        probability for each class
    """
    proba = np.zeros(no_classes)
    unique_classes, count_unique_classes = np.unique(y, return_counts=True)
    proba[unique_classes] = count_unique_classes/len(y)
    return proba


@njit(fastmath=True)
def get_potential_splits(X, y):
    potential_splt = list()
    for col_index in range(X.shape[1]):
        potential_splt.append(np.unique(X[:, col_index]))
    return potential_splt


@njit("(f8[:, :], i8[:], f8[:], i8)", fastmath=True)
def split_data(X, y, beta, split_value):
    """
    split data so that Xbeta<0 goes to the left note and Xbeta>0 goes to the right node
    """
    produs = np_dot_numba(X, beta)
    data_below = X[produs <= split_value]
    data_above = X[produs > split_value]
    y_below = y[produs <= split_value]
    y_above = y[produs > split_value]
    return data_below, data_above, y_below, y_above


def calculate_overall_entropy(n_points, X_below_len, X_above_len, y_below, y_above):
    p_below = X_below_len / n_points
    p_above = X_above_len / n_points
    return (p_below * entropy_numba(y_below)) + (p_above * entropy_numba(y_above))


def determine_best_split_joc(X, y, params_joc):
    """

    :param X:
    :param y:
    :param params_joc: game parameters (named tuple) with keys "zg_normal", "devieri", "mu", "sigma", "max_depth", "pop_size"

    :return: beta for all attributes 
    """
    beta = norm.rvs(size=X.shape[1] * 2)

    for _ in range(params_joc.pop_size):
        v, beta = get_v(beta, X, y, params_joc.devieri*X.shape[1], params_joc.mu,
                        params_joc.sigma, params_joc.zg_normal)

    marime_beta = X.shape[1]
    beta_left = beta[:marime_beta]

    beta_right = beta[marime_beta:]

    beta_nou = np.zeros(marime_beta)

    for i in range(marime_beta):
        beta_nou[i] = (beta_left[i] + beta_right[i]) / 2.0

    return beta_nou, 0


def get_v(Beta, X, y, nr_devieri, loc_random, scale_random, zgomot_normal):
    """optimization of v(.) function"""
    marime_beta = X.shape[1]
    payoffs = payoff_toti_numba(X, y, Beta)
    v = 0
    Beta_diferente_maxime = Beta.copy()
    diferente_maxime = np.zeros(2)
    for jucator in range(2):
        payoff_nou = np.zeros(nr_devieri)
        for i in range(nr_devieri):
            _beta = Beta.copy()
            for j in range(marime_beta):
                if zgomot_normal:
                    _beta[jucator * marime_beta + j] = Beta[
                        jucator * marime_beta + j] + np.random.normal(
                        loc=loc_random, scale=scale_random)
                else:
                    _beta[jucator * marime_beta + j] = Beta[
                        jucator * marime_beta + j] + np.random.uniform(loc_random, scale_random)
            payoff_nou[i] = payoff_nod_numba(X, y, _beta, jucator)
            if payoffs[jucator]-payoff_nou[i] > diferente_maxime[jucator]:
                diferente_maxime[jucator] = payoffs[jucator]-payoff_nou[i]
                for j in range(marime_beta):
                    Beta_diferente_maxime[jucator * marime_beta +
                                          j] = _beta[jucator * marime_beta + j]

        diferente = payoffs[jucator] - payoff_nou
        v += np.sum(diferente[diferente > 0] ** 2)
    return v, Beta_diferente_maxime


def dt_alg(X, y, params_joc, counter=0, min_samples=2, max_depth=5, no_classes=3):
    """basic decision tree algorithm"""
    if (check_purity(y)) or (len(X) < min_samples) or (counter == max_depth):
        leaf = create_leaf(X, y, no_classes)
        return leaf, 0
    else:
        counter += 1
        beta, split_value = determine_best_split_joc(X, y, params_joc)

        X_below, X_above, y_below, y_above = split_data(
            X, y, beta, split_value)

        if (len(X_below) == 0) or (len(X_above) == 0):
            leaf = create_leaf(X, y, no_classes)
            return leaf, 0

        question = f'{",".join([str(b) for b in beta])} $<=$ {split_value}'
        sub_tree = {question: []}
        yes_answer = dt_alg(X_below, y_below, params_joc,
                            counter, min_samples, max_depth, no_classes)
        no_answer = dt_alg(X_above, y_above, params_joc,
                           counter, min_samples, max_depth, no_classes)

        if isinstance(yes_answer, tuple) and isinstance(no_answer, tuple):
            if np.all(yes_answer[0] == no_answer[0]):
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
        else:
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

        return sub_tree


def classify_example(example, tree):
    """classify an example"""
    question = list(tree.keys())[0]
    beta, comparison_operator, value = question.split(sep='$')
    beta = [float(b) for b in list(beta.split(","))]
    if np.dot(beta, example) <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    if not isinstance(answer, dict):
        if isinstance(answer, tuple):
            if isinstance(answer[1], np.ndarray):
                if np.isnan(ndtrVec_numba(np.dot(example, answer[1]))):
                    print('am un nan, intorc probabilitatea si nu ndtr()')
                    return answer[0]
                return ndtrVec_numba(np.dot(example, answer[1]))
            else:
                # all labels from the leaf node are from the same class
                return answer[0]
        else:
            return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def predict(X, tree):
    """predict labels for X"""
    pred = list()
    pred_clase = list()
    for el in X:
        pred.append(classify_example(el, tree))
        pred_clase.append(np.argmax(pred[-1]))
    return pred, pred_clase
