# pmas/agents/explain_agent.py
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure plots save without needing a display
import matplotlib.pyplot as plt
import os

class ExplainAgent:
    """
    SHAP explanation agent.
    Supports:
      - Standard models stored as {"scaler": scaler, "clf": clf}
      - Fairlearn ExponentiatedGradient (unwraps best classifier from ensemble)
    Saves SHAP summary plot to outputs/shap_summary.png
    """

    def __init__(self, output_path="outputs/shap_summary.png"):
        self.output_path = output_path
        self.FEATURES = ["pregnant","glucose","pressure","triceps",
                         "insulin","mass","pedigree","age"]

    def _unwrap_fairlearn_expgrad(self, clf):
        """Unwrap Fairlearn ExponentiatedGradient to a usable sklearn classifier."""
        # Newer Fairlearn: predictors_ and weights_
        if hasattr(clf, "predictors_") and hasattr(clf, "weights_"):
            predictors = list(clf.predictors_)
            weights = np.array(clf.weights_)
            if len(predictors) > 0 and len(predictors) == len(weights):
                best_idx = int(np.argmax(weights))
                chosen = predictors[best_idx]
                print(f"[ExplainAgent] Using predictor #{best_idx} with weight={weights[best_idx]:.3f} from ExponentiatedGradient")
                return chosen
        # Older versions: best_classifier_
        if hasattr(clf, "best_classifier_"):
            print("[ExplainAgent] Using best_classifier_ from ExponentiatedGradient")
            return clf.best_classifier_
        raise RuntimeError("Could not unwrap ExponentiatedGradient: no predictors_ or best_classifier_")

    def perform(self, action, params, state=None):
        if action != "shap":
            raise NotImplementedError(action)

        model = state.get("model")
        if model is None:
            raise RuntimeError("model not found in state")

        # unpack dict {"scaler":..., "clf":...}
        if isinstance(model, dict) and "clf" in model:
            clf = model["clf"]
            scaler = model.get("scaler", None)
        else:
            raise RuntimeError("Unexpected model format for SHAP")

        # unwrap Fairlearn ExponentiatedGradient
        if clf.__class__.__name__ == "ExponentiatedGradient":
            clf = self._unwrap_fairlearn_expgrad(clf)

        train = state.get("train_df")
        test = state.get("test_df")
        if train is None or test is None:
            raise RuntimeError("train_df or test_df not found in state")

        Xtrain = train[self.FEATURES].values
        Xtest = test[self.FEATURES].values

        # scale if available
        if scaler is not None:
            Xtrain_scaled = scaler.transform(Xtrain)
            Xtest_scaled = scaler.transform(Xtest)
        else:
            Xtrain_scaled, Xtest_scaled = Xtrain, Xtest

        # Try LinearExplainer first
        try:
            explainer = shap.LinearExplainer(clf, Xtrain_scaled)
            shap_values = explainer.shap_values(Xtest_scaled)
        except Exception as e:
            # fallback: KernelExplainer (slower, approximate)
            try:
                print("LinearExplainer failed, using KernelExplainer fallback:", e)
                f = (lambda x: clf.predict(x)) if hasattr(clf, "predict") else None
                if f is None:
                    raise RuntimeError("Model has no predict method")
                explainer = shap.KernelExplainer(f, shap.sample(Xtrain_scaled, min(50, len(Xtrain_scaled))))
                shap_values = explainer.shap_values(Xtest_scaled[:50])
                Xtest_scaled = Xtest_scaled[:50]
            except Exception as e2:
                raise RuntimeError(f"SHAP explainer failed: {e} | fallback error: {e2}")

        # Save SHAP summary plot
        plt.figure(figsize=(6, 4))
        try:
            shap.summary_plot(shap_values, Xtest_scaled,
                              feature_names=self.FEATURES, show=False)
        except Exception:
            if isinstance(shap_values, (list, tuple)):
                shap.summary_plot(shap_values[0], Xtest_scaled,
                                  feature_names=self.FEATURES, show=False)
            else:
                raise
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=150)
        plt.close()

        return {"shap_path": self.output_path}

