import copy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from statistics import mean, stdev
import math
import numpy as np
import gc
from warnings import filterwarnings

filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from .LogisticRegression_update import LogisticRegression_update

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)

from scipy.sparse import csr_matrix

def train_lr(   train_x,
                train_y,
                random_state,
                soft_label=False,
                ):

    if soft_label:
        clf = LogisticRegression_update(random_state = random_state, max_iter=100)
        y_hard = np.argmax(train_y, axis=1)
        if len(y_hard.shape) > 1:
            y_hard = np.ravel(y_hard)
            temp = csr_matrix(train_y)
            train_y = temp.toarray()
        clf.fit(train_x, y_hard, train_y)
        return clf
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(random_state = random_state, max_iter=100))
        clf.fit(train_x, train_y)
        return clf


def evaluate_svm(   clf,
                    test_x,
                    test_y,
                    soft_label = False,
                    ):
    if soft_label:
        test_y_pred = clf.predict(test_x)
        acc = accuracy_score(test_y, test_y_pred)
    else:
        test_y_pred = clf.predict(test_x)
        acc = accuracy_score(test_y, test_y_pred)
    return acc


def train_eval_lr(  train_x, train_y,
                    test_x, test_y,
                    num_seeds = 3,
                    soft_label = False,
                    fewshot = False,
                    ):
    train_acc_list = []
    reg_acc_list = []

    for random_state in range(num_seeds):
        clf_reg = train_lr(train_x, train_y, random_state, soft_label)
        train_y = np.argmax(train_y, axis=1) if soft_label else train_y
        test_y = test_y / 2 if fewshot and soft_label else test_y
        train_acc_list.append(evaluate_svm(clf_reg, train_x, train_y, soft_label)) # np.argmax(train_y, axis=1)
        reg_acc_list.append(evaluate_svm(clf_reg, test_x, test_y, soft_label))

    if num_seeds > 1:
        deviation = stdev(reg_acc_list)
        if deviation > 0.01:
            print(f"big stdev! {reg_acc_list}")

    return mean(train_acc_list), mean(reg_acc_list)

