# pmas/agents/model_agent.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate

class ModelAgent:
    """
    Train & evaluate. Feature list is read from state['features'] if present.
    """
    def __init__(self):
        self.scaler = None
        self.clf = None

    def _fit(self, X, y, sample_weight=None):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.clf = LogisticRegression(solver="liblinear", max_iter=200)
        if sample_weight is not None:
            self.clf.fit(Xs, y, sample_weight=sample_weight)
        else:
            self.clf.fit(Xs, y)
        return {"model": {"scaler": self.scaler, "clf": self.clf}}

    def _unpack_model(self, model, state):
        # dict with scaler/clf
        if isinstance(model, dict) and "scaler" in model and "clf" in model:
            return model["scaler"], model["clf"]
        # sklearn Pipeline-like
        try:
            named = getattr(model, "named_steps", None)
            if named and "scaler" in named and "clf" in named:
                return named["scaler"], named["clf"]
        except Exception:
            pass
        # estimator object
        if hasattr(model, "predict"):
            scaler = state.get("model", {}).get("scaler", None) or self.scaler
            return scaler, model
        raise RuntimeError("Unrecognized model format")

    def perform(self, action, params, state=None):
        if state is None:
            state = {}
        features = state.get("features")
        if action == "train":
            train_df = state.get("train_df")
            if train_df is None or features is None:
                raise RuntimeError("train_df or features not found")
            X = train_df[features].values
            y = train_df["y"].values
            sample_weight = params.get("sample_weight", None)
            return self._fit(X, y, sample_weight=sample_weight)

        if action == "eval":
            test_df = state.get("test_df")
            if test_df is None or features is None:
                raise RuntimeError("test_df or features not found")
            model_in_state = state.get("model")
            model = model_in_state if model_in_state is not None else {"scaler": self.scaler, "clf": self.clf}
            scaler, clf = self._unpack_model(model, state)
            Xtest = test_df[features].values
            Xtest_scaled = scaler.transform(Xtest) if scaler is not None else Xtest
            ytrue = test_df["y"].values
            ypred = clf.predict(Xtest_scaled)
            metrics = {
                "accuracy": float(accuracy_score(ytrue, ypred)),
                "precision": float(precision_score(ytrue, ypred, zero_division=0)),
                "recall": float(recall_score(ytrue, ypred, zero_division=0))
            }
            sensitive_name = state.get("sensitive_name", "age_group")
            sensitive = test_df[sensitive_name]
            metric_frame = MetricFrame(metrics={"accuracy": accuracy_score,
                                               "fpr": false_positive_rate,
                                               "fnr": false_negative_rate},
                                       y_true=ytrue,
                                       y_pred=ypred,
                                       sensitive_features=sensitive)
            group_metrics = metric_frame.by_group.to_dict()
            return {"y_pred": ypred, "eval_metrics": metrics, "group_metrics": group_metrics}
        raise NotImplementedError(action)
