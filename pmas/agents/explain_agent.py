# pmas/agents/explain_agent.py
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

class ExplainAgent:
    def __init__(self, output_path="outputs/shap_summary.png"):
        self.output_path = output_path

    def _unwrap_fairlearn_expgrad(self, clf):
        if hasattr(clf, "predictors_") and hasattr(clf, "weights_"):
            predictors = list(clf.predictors_)
            weights = np.array(clf.weights_)
            if len(predictors) > 0 and len(predictors) == len(weights):
                best_idx = int(np.argmax(weights))
                chosen = predictors[best_idx]
                print(f"[ExplainAgent] Using predictor #{best_idx} with weight={weights[best_idx]:.3f} from ExponentiatedGradient")
                return chosen
        if hasattr(clf, "best_classifier_"):
            print("[ExplainAgent] Using best_classifier_ from ExponentiatedGradient")
            return clf.best_classifier_
        raise RuntimeError("Could not unwrap ExponentiatedGradient")

    def perform(self, action, params, state=None):
        if action != "shap":
            raise NotImplementedError(action)
        model = state.get("model")
        if model is None:
            raise RuntimeError("model not found")
        if isinstance(model, dict) and "clf" in model:
            clf = model["clf"]
            scaler = model.get("scaler", None)
        else:
            raise RuntimeError("Unexpected model format for SHAP")
        # unwrap if needed
        if clf.__class__.__name__ == "ExponentiatedGradient":
            clf = self._unwrap_fairlearn_expgrad(clf)
        train = state.get("train_df")
        test = state.get("test_df")
        features = state.get("features")
        if train is None or test is None or features is None:
            raise RuntimeError("train/test/features required for SHAP")
        Xtrain = train[features].values
        Xtest = test[features].values
        if scaler is not None:
            Xtrain_scaled = scaler.transform(Xtrain)
            Xtest_scaled = scaler.transform(Xtest)
        else:
            Xtrain_scaled, Xtest_scaled = Xtrain, Xtest
        try:
            explainer = shap.LinearExplainer(clf, Xtrain_scaled)
            shap_values = explainer.shap_values(Xtest_scaled)
        except Exception as e:
            try:
                print("LinearExplainer failed, using KernelExplainer fallback:", e)
                f = (lambda x: clf.predict(x)) if hasattr(clf, "predict") else None
                if f is None:
                    raise RuntimeError("Model has no predict")
                explainer = shap.KernelExplainer(f, shap.sample(Xtrain_scaled, min(50, len(Xtrain_scaled))))
                shap_values = explainer.shap_values(Xtest_scaled[:50])
                Xtest_scaled = Xtest_scaled[:50]
            except Exception as e2:
                raise RuntimeError(f"SHAP explainer failed: {e} | fallback error: {e2}")
        plt.figure(figsize=(6,4))
        try:
            shap.summary_plot(shap_values, Xtest_scaled, feature_names=features, show=False)
        except Exception:
            if isinstance(shap_values, (list, tuple)):
                shap.summary_plot(shap_values[0], Xtest_scaled, feature_names=features, show=False)
            else:
                raise
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=150)
        plt.close()
        return {"shap_path": self.output_path}
