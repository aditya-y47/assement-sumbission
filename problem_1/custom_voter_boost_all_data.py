import os
from pprint import pprint
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from predictors import LightGBMClassifier, WeightedEnsembleClassifier, XGBoostClassifier


def compute_metrics(y_pred, y_test):
    # y_pred = clf.predict(X_test, threshold=None)
    # probas = clf.predict_proba(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_dict = {"precision": precision, "recall": recall, "f1_score": f1}

    return metrics_dict


data = pd.read_csv("./clean_data.csv")
targets = data.pop(list(data.columns)[-1])

X_train, X_test, y_train, y_test = train_test_split(
    data, targets, test_size=0.3, random_state=121, stratify=targets
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.67, random_state=121, stratify=y_test
)
data = {}
data["train"] = [X_train, y_train]
data["val"] = [X_val, y_val]
data["test"] = [X_test, y_test]


unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
class_weights = {}
total_samples = np.sum(counts)
for cls in class_counts:
    class_weights[cls] = total_samples / (len(class_counts) * class_counts[cls])
sample_weights = np.array([class_weights[cls] for cls in y_train])

xgb_list = []
xgb_classifiers = []
lgbm_list = []
lgbm_classifiers = []

for root, dirs, files in os.walk("params"):
    for file in files:
        if file.endswith("json") and "xgb" in file:
            xgb_list.append(os.path.join(root, file))

for i in range(len(xgb_list)):
    _params = json.load(open(xgb_list[i]))
    xgb_classifiers.append(XGBoostClassifier(params=_params, data=data))

for xgb_classifier in xgb_classifiers:
    xgb_classifier.fit(None, None)

for root, dirs, files in os.walk("params/all_data"):
    for file in files:
        if file.endswith("json") and "lgbm" in file:
            lgbm_list.append(os.path.join(root, file))

for i in range(len(lgbm_list)):
    _params = json.load(open(lgbm_list[i]))
    lgbm_classifiers.append(LightGBMClassifier(params=_params, data=data))

for lgbm_classifier in lgbm_classifiers:
    lgbm_classifier.fit(None, None)

all_estimators = xgb_classifiers + lgbm_classifiers
