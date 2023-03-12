import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

warnings.filterwarnings(
    "ignore",
    message="[LightGBM] \[Warning\] No further splits with positive gain, best gain: -inf",
)

tuned_params_xgb = "./params/tuned_params_xgb.json"
tuned_params_lgbm = "./params/tuned_params_lgbm.json"


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, params=None, num_boost_round=1000, early_stopping_rounds=10, data=None
    ):
        if params is None:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "n_jobs": -1,
            }
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.data = data
        self.threshold = None

    def fit(self, X, y):
        if X is None:
            data = self.data
            X = data["train"][0]
            y = data["train"][1]
        X_val = data["val"][0]
        y_val = data["val"][1]
        scale_pos_weight = (len(y) - y.sum()) / y.sum()
        dtrain = xgb.DMatrix(X, label=y)
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
        self.params["scale_pos_weight"] = scale_pos_weight

        early_stop = xgb.callback.EarlyStopping(
            rounds=self.early_stopping_rounds,
            save_best=False,
            maximize=False,
        )
        eval_set = [(dtrain, "train"), (dval, "validation")]
        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=10,
            callbacks=[early_stop],
        )
        y_train_probs = self.booster.predict(dtrain)
        fpr, tpr, thresholds = roc_curve(y, y_train_probs)
        self.threshold = thresholds[np.argmax(tpr - fpr)]

        return self

    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        return self.booster.predict(dmat)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Model has not been fit yet")
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        return y_pred

    def score(self, X, y):
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc_roc = roc_auc_score(y, y_pred_proba)
        auc_pr = average_precision_score(y, y_pred_proba)
        accuracy = accuracy_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics_dict = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "accuracy": accuracy,
            "auc_pr": auc_pr,
            "threshold": self.threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        return metrics_dict


class LightGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, params=None, num_boost_round=1000, early_stopping_rounds=10, data=None
    ):
        if params is None:
            params = {
                "objective": "binary",
                "metric": "aucpr",
                "boosting_type": "gbdt",
            }
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None
        self.data = data
        self.threshold = None

    def fit(self, X, y):
        if X is None:
            X = self.data["train"][0]
            y = self.data["train"][1]
        scale_pos_weight = (len(y) - y.sum()) / y.sum()
        train_data = lgb.Dataset(X, label=y)

        # Pass validation data to the train method as a dictionary
        evals_result = {}
        if self.data.get("val") is not None:
            val_data = lgb.Dataset(self.data["val"][0], self.data["val"][1])
            evals_result = {
                "eval_set": [(self.data["val"][0], self.data["val"][1])],
                "eval_metric": self.params["metric"],
            }

        self.params["scale_pos_weight"] = scale_pos_weight

        self.booster = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_data, val_data],
            verbose_eval=False,
            evals_result=evals_result,
        )
        y_train_proba = self.booster.predict(
            X, num_iteration=self.booster.best_iteration
        )
        fpr, tpr, thresholds = roc_curve(y, y_train_proba)
        self.threshold = thresholds[np.argmax(tpr - fpr)]

        return self

    def predict_proba(self, X):
        return self.booster.predict(X)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("The model has not been fit yet.")
        y_proba = self.predict_proba(X)
        return (y_proba > self.threshold).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc_roc = roc_auc_score(y, y_pred)
        auc_pr = average_precision_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics_dict = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": accuracy,
            "threshold": self.threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        return metrics_dict


class WeightedEnsembleClassifier:
    """
    A weighted ensemble classifier based on AUC-PR scores of individual classifiers.
    """

    def __init__(self, estimators, weights=None):
        """
        Constructor for the WeightedEnsembleClassifier class."""
        self.estimators = estimators
        if weights is not None:
            self.weights = weights
        else:
            self.weights = None

    def score(self, X, y):
        scores = []
        for estimator in self.estimators:
            if not hasattr(estimator, "score"):
                raise ValueError(
                    "Some estimators in the ensemble do not have a 'score' method"
                )
            score = estimator.score(X, y)
            scores.append(score)

        scores = pd.DataFrame(scores)
        weights = (
            self.weights
            if self.weights is not None
            else np.ones(len(scores)) / len(scores)
        )
        weights = weights * scores["auc_pr"] / np.sum(weights * scores["auc_pr"])
        threshold = np.mean(scores["threshold"])
        return weights.values, threshold

    def predict(self, X, threshold):
        if not all(
            [hasattr(estimator, "predict_proba") for estimator in self.estimators]
        ):
            raise ValueError("Some estimators in the ensemble have not been fit")

        if self.weights is None:
            probas = np.mean(
                [estimator.predict_proba(X) for estimator in self.estimators],
                axis=0,
            )
        else:
            probas = np.average(
                [estimator.predict_proba(X) for estimator in self.estimators],
                axis=0,
                weights=self.weights,
            )

        # Use the provided threshold
        if threshold is None:
            raise ValueError("Threshold value for Prediction not provided")

        # Make predictions based on whether the probability of the positive class exceeds the threshold
        y_pred = (probas > threshold).astype(int)

        return y_pred
