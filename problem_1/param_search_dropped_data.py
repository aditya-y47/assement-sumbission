import json

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from predictors import LightGBMClassifier, XGBoostClassifier

data = pd.read_csv("./clean_data_dropped.csv")
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


def objective_xgb(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_jobs": -1,
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-6, 1e2),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-6, 1e2),
        "gamma": trial.suggest_loguniform("gamma", 1e-6, 1e1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
        "tree_method": "hist",
    }

    clf = XGBoostClassifier(
        params=params, data={"train": (X_train, y_train), "val": (X_val, y_val)}
    )
    clf.fit(X=None, y=None)
    return 1 - clf.score(data["val"][0], data["val"][1])["auc_pr"]


default_params = XGBoostClassifier().get_params()["params"]
study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=500)
count = 1
top_trials = study.trials_dataframe().sort_values(by="value").iloc[:3, :]
for i, trial in top_trials.iterrows():
    default_params.update(
        {
            "colsample_bytree": trial["params_colsample_bytree"],
            "gamma": trial["params_gamma"],
            "learning_rate": trial["params_learning_rate"],
            "max_depth": trial["params_max_depth"],
            "min_child_weight": trial["params_min_child_weight"],
            "reg_alpha": trial["params_reg_alpha"],
            "reg_lambda": trial["params_reg_lambda"],
            "subsample": trial["params_subsample"],
        }
    )
    with open(f"params/dropped_data/params_tuned_xgb_{count}.json", "w") as f:
        json.dump(default_params, f, indent=4, sort_keys=True)
    count += 1


def objective_lgbm(trial):
    params = {
        "objective": "binary",
        "metric": "aucpr",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
    }
    clf_2 = LightGBMClassifier(
        params=params, data={"train": (X_train, y_train), "val": (X_val, y_val)}
    )
    clf_2.fit(X=None, y=None)
    return 1 - clf_2.score(data["val"][0], data["val"][1])["auc_pr"]


study_lgbm = optuna.create_study(direction="minimize")
study_lgbm.optimize(objective_lgbm, n_trials=500, show_progress_bar=True)
default_params = LightGBMClassifier().get_params()["params"]


top_trials = study_lgbm.trials_dataframe().sort_values(by="value").iloc[:3, :]
count = 1
for i, trial in top_trials.iterrows():
    default_params.update(
        {
            "colsample_bytree": trial["params_colsample_bytree"],
            "learning_rate": trial["params_learning_rate"],
            "max_depth": trial["params_max_depth"],
            "min_child_samples": trial["params_min_child_samples"],
            "n_estimators": trial["params_n_estimators"],
            "num_leaves": trial["params_num_leaves"],
            "reg_alpha": trial["params_reg_alpha"],
            "reg_lambda": trial["params_reg_lambda"],
            "subsample": trial["params_subsample"],
        }
    )
    with open(f"params/dropped_data/params_tuned_lgbm_{count}.json", "w") as f:
        json.dump(default_params, f, indent=4, sort_keys=True)
    count += 1
