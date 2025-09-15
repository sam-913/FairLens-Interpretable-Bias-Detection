# pmas/agents/mitigation_agent.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Fairlearn reductions (for expgrad)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

class MitigationAgent:
    """
    Two mitigation strategies:
      - "reweight": simple sample-weight reweighting (demonstrative)
      - "expgrad": Fairlearn ExponentiatedGradient (reduction) producing a fitted mitigated predictor.

    For compatibility with the rest of the pipeline, the expgrad branch returns both:
      - "exp_model": {"scaler": scaler, "clf": mitigator}
      - "model": {"scaler": scaler, "clf": mitigator}
    so existing code that expects "exp_model" will keep working and ModelAgent can also
    pick up "model" directly.
    """
    def __init__(self):
        pass

    def perform(self, action, params, state=None):
        if action == "reweight":
            train = state.get("train_df")
            if train is None:
                raise RuntimeError("train_df not found")
            grouped = train.groupby("age_group")["y"].agg(["sum", "count"])
            overall_pos = train["y"].mean()
            weights = np.ones(len(train), dtype=float)
            for g, row in grouped.iterrows():
                group_pos_rate = (row["sum"] / row["count"]) if row["count"] > 0 else 0.0
                if group_pos_rate == 0:
                    multiplier = 1.0
                else:
                    multiplier = overall_pos / group_pos_rate
                weights[train["age_group"] == g] = multiplier
            weights = weights / np.mean(weights)
            return {"sample_weight": weights}

        if action == "expgrad":
            # Fit ExponentiatedGradient (DemographicParity) on scaled features
            train = state.get("train_df")
            if train is None:
                raise RuntimeError("train_df not found")
            FEATURES = ["pregnant","glucose","pressure","triceps","insulin","mass","pedigree","age"]
            X = train[FEATURES].values
            y = train["y"].values
            sensitive = train["age_group"].values

            # scale features (same scaling will be used at eval time)
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)

            base_est = LogisticRegression(solver="liblinear", max_iter=200)

            # fairlearn>=0.12 uses 'estimator' keyword
            mitigator = ExponentiatedGradient(estimator=base_est, constraints=DemographicParity())
            mitigator.fit(Xs, y, sensitive_features=sensitive)

            model_dict = {"scaler": scaler, "clf": mitigator}
            # return both keys for backward compatibility
            return {"exp_model": model_dict, "model": model_dict}

        raise NotImplementedError(action)
