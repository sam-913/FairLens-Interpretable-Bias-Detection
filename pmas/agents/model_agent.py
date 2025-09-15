# pmas/agents/model_agent.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate
from typing import Any

class ModelAgent:
    """
    Train a logistic regression classifier on the provided train_df and evaluate on test_df.
    Uses sample_weight if provided in params (for mitigation path).

    Updated eval to accept multiple model shapes:
      - dict {"scaler": scaler, "clf": clf}
      - sklearn Pipeline-like objects with .named_steps
      - direct estimator objects (with predict method); if no scaler present, will try to use
        a scaler fitted from train_df (if available in state) for consistent scaling.
    """
    FEATURES = ["pregnant","glucose","pressure","triceps","insulin","mass","pedigree","age"]

    def __init__(self):
        # hold fitted scaler+clf when trained via this agent
        self.scaler = None
        self.clf = None
        self.pipeline = None  # legacy / optional

    def _fit(self, X, y, sample_weight=None):
        # fit scaler
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        # fit classifier
        self.clf = LogisticRegression(solver="liblinear", max_iter=200)
        if sample_weight is not None:
            self.clf.fit(Xs, y, sample_weight=sample_weight)
        else:
            self.clf.fit(Xs, y)
        # store a simple pipeline-like container
        self.pipeline = {"scaler": self.scaler, "clf": self.clf}

    def _unpack_model(self, model: Any, state: dict):
        """
        Normalize different model representations into (scaler, clf) pair.
        Returns (scaler, clf).
        """
        # 1) If model is a dict with scaler/clf
        if isinstance(model, dict):
            if "scaler" in model and "clf" in model:
                return model["scaler"], model["clf"]
            # some versions might nest under "model" or "exp_model" - handled by caller before passing
        # 2) If model is a sklearn Pipeline-like object with named_steps
        try:
            named = getattr(model, "named_steps", None)
            if named and "clf" in named and "scaler" in named:
                return named["scaler"], named["clf"]
        except Exception:
            pass
        # 3) If model is an estimator object (has predict), try to find scaler in state or use self.scaler
        if hasattr(model, "predict"):
            # attempt to get scaler from state (maybe mitigation returned scaler separately)
            s = state.get("model", None)
            if isinstance(s, dict) and "scaler" in s:
                return s["scaler"], model
            # fallback to self.scaler if trained earlier
            if self.scaler is not None:
                return self.scaler, model
            # fallback: no scaler available — return (None, model) and caller must handle it
            return None, model

        raise RuntimeError("Unrecognized model format in ModelAgent._unpack_model")

    def perform(self, action, params, state=None):
        if action == "train":
            train_df = state.get("train_df")
            if train_df is None:
                raise RuntimeError("train_df not found in state")
            X = train_df[self.FEATURES].values
            y = train_df["y"].values
            sample_weight = params.get("sample_weight", None)
            self._fit(X, y, sample_weight=sample_weight)
            # return model representation so other agents can use it
            return {"model": {"scaler": self.scaler, "clf": self.clf}, "pipeline": self.pipeline}
        if action == "eval":
            test_df = state.get("test_df")
            if test_df is None:
                raise RuntimeError("test_df not found in state")
            # model may be passed in state (from mitigation) or use the one trained here
            model_in_state = state.get("model")
            model = model_in_state if model_in_state is not None else ({"scaler": self.scaler, "clf": self.clf} if self.clf is not None else None)
            if model is None:
                raise RuntimeError("No model available for evaluation")
            scaler, clf = self._unpack_model(model, state=state or {})
            # prepare test features; if scaler is present use it, else use raw features
            Xtest = test_df[self.FEATURES].values
            if scaler is not None:
                Xtest_scaled = scaler.transform(Xtest)
            else:
                Xtest_scaled = Xtest
            # prediction (clf is expected to implement predict)
            ytrue = test_df["y"].values
            ypred = clf.predict(Xtest_scaled)
            # overall metrics
            metrics = {
                "accuracy": float(accuracy_score(ytrue, ypred)),
                "precision": float(precision_score(ytrue, ypred, zero_division=0)),
                "recall": float(recall_score(ytrue, ypred, zero_division=0))
            }
            # fairness metrics across 'age_group'
            sensitive = test_df["age_group"]
            metric_frame = MetricFrame(metrics={"accuracy": accuracy_score,
                                               "fpr": false_positive_rate,
                                               "fnr": false_negative_rate},
                                       y_true=ytrue,
                                       y_pred=ypred,
                                       sensitive_features=sensitive)
            group_metrics = metric_frame.by_group.to_dict()
            return {"y_pred": ypred, "eval_metrics": metrics, "group_metrics": group_metrics}
        raise NotImplementedError(action)

