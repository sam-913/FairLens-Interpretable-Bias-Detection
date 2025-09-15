# pmas/main.py
import logging
from pmas.orchestrator import Orchestrator
from pmas.agents.data_agent import DataAgent
from pmas.agents.model_agent import ModelAgent
from pmas.agents.explain_agent import ExplainAgent
from pmas.agents.mitigation_agent import MitigationAgent
import os

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def run_full_pipeline(use_mitigation=False, mitigation_method="reweight"):
    """
    mitigation_method: "reweight" (simple) or "expgrad" (Fairlearn ExponentiatedGradient)
    """
    setup_logging()
    orch = Orchestrator()

    data_agent = DataAgent()
    model_agent = ModelAgent()
    explain_agent = ExplainAgent()
    mitigation_agent = MitigationAgent()

    orch.register("data", data_agent)
    orch.register("model", model_agent)
    orch.register("explain", explain_agent)
    orch.register("mitig", mitigation_agent)

    # initial run
    steps = [
        {"agent":"data","action":"load"},
        {"agent":"model","action":"train","params":{}},
        {"agent":"model","action":"eval","params":{}},
        {"agent":"explain","action":"shap","params":{}}
    ]
    state = orch.run(steps)

    # store baseline metrics for comparison
    baseline_metrics = state.get("eval_metrics", {})
    baseline_group_metrics = state.get("group_metrics", {})

    if use_mitigation:
        if mitigation_method == "reweight":
            # compute weights using mitigation agent and retrain
            m_state = mitigation_agent.perform("reweight", {}, state=state)
            sample_weight = m_state.get("sample_weight")
            retrain_steps = [
                {"agent":"model","action":"train","params":{"sample_weight": sample_weight}},
                {"agent":"model","action":"eval","params":{}},
                {"agent":"explain","action":"shap","params":{}}
            ]
            state = orch.run(retrain_steps, initial_state=state)
        elif mitigation_method == "expgrad":
            # compute ExponentiatedGradient model and evaluate it (no re-train via model_agent)
            m_state = mitigation_agent.perform("expgrad", {}, state=state)
            exp_model = m_state.get("exp_model")
            # place exp_model into state as 'model' so ModelAgent.eval can evaluate it
            state_with_exp = dict(state)
            state_with_exp["model"] = exp_model
            # evaluate and explain using exp_model (no train)
            eval_steps = [
                {"agent":"model","action":"eval","params":{}},
                {"agent":"explain","action":"shap","params":{}}
            ]
            state = orch.run(eval_steps, initial_state=state_with_exp)
        else:
            raise ValueError("Unknown mitigation_method: " + str(mitigation_method))

    # attach baseline metrics for comparison
    state["baseline_metrics"] = baseline_metrics
    state["baseline_group_metrics"] = baseline_group_metrics

    os.makedirs("outputs", exist_ok=True)
    return state

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    s = run_full_pipeline(use_mitigation=False)
    print("Baseline metrics:", s.get("eval_metrics"))
    print("Group metrics:", s.get("group_metrics"))
    print("SHAP image:", s.get("shap_path"))

