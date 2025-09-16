# pmas/agents/mitigation_agent.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import ExponentiatedGradient, DemographicParity

class MitigationAgent:
    def __init__(self):
        pass

    def perform(self, action, params, state=None):
        if action == "reweight":
            train = state.get("train_df")
            sensitive_name = state.get("sensitive_name", "age_group")
            if train is None:
                raise RuntimeError("train_df not found")
            grouped = train.groupby(sensitive_name)["y"].agg(["sum", "count"])
            overall_pos = train["y"].mean()
            weights = np.ones(len(train), dtype=float)
            for g, row in grouped.iterrows():
                group_pos_rate = (row["sum"] / row["count"]) if row["count"] > 0 else 0.0
                multiplier = 1.0 if group_pos_rate == 0 else (overall_pos / group_pos_rate)
                weights[train[sensitive_name] == g] = multiplier
            weights = weights / np.mean(weights)
            return {"sample_weight": weights}

        if action == "expgrad":
            train = state.get("train_df")
            sensitive_name = state.get("sensitive_name", "age_group")
            FEATURES = state.get("features")
            if train is None or FEATURES is None:
                raise RuntimeError("train_df or features not found")
            X = train[FEATURES].values
            y = train["y"].values
            sensitive = train[sensitive_name].values

            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)

            base_est = LogisticRegression(solver="liblinear", max_iter=200)
            mitigator = ExponentiatedGradient(estimator=base_est, constraints=DemographicParity())
            mitigator.fit(Xs, y, sensitive_features=sensitive)

            model_dict = {"scaler": scaler, "clf": mitigator}
            return {"exp_model": model_dict, "model": model_dict}

        raise NotImplementedError(action)
