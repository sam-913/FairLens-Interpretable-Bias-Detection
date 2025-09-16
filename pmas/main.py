# pmas/main.py
import logging
import os
from pmas.orchestrator import Orchestrator
from pmas.agents.data_agent import DataAgent
from pmas.agents.model_agent import ModelAgent
from pmas.agents.explain_agent import ExplainAgent
from pmas.agents.mitigation_agent import MitigationAgent

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def run_full_pipeline(use_mitigation=False,
                      mitigation_method="reweight",
                      dataset="Pima Diabetes",
                      use_cache_only=False):
    """
    Run full pipeline and return final state.

    Parameters
    ----------
    use_mitigation : bool
        whether to run mitigation after baseline
    mitigation_method : str
        "reweight" or "expgrad"
    dataset : str
        "Pima Diabetes" or "Adult Income"
    use_cache_only : bool
        If True, the data loader will not attempt downloads and requires cached files.
    """
    setup_logging()
    orch = Orchestrator()

    # instantiate agents
    data_agent = DataAgent()
    model_agent = ModelAgent()
    explain_agent = ExplainAgent()
    mitigation_agent = MitigationAgent()

    orch.register("data", data_agent)
    orch.register("model", model_agent)
    orch.register("explain", explain_agent)
    orch.register("mitig", mitigation_agent)

    # pass dataset + use_cache_only into the load step
    steps = [
        {"agent": "data", "action": "load", "params": {"dataset": dataset, "use_cache_only": use_cache_only}},
        {"agent": "model", "action": "train", "params": {}},
        {"agent": "model", "action": "eval", "params": {}},
        {"agent": "explain", "action": "shap", "params": {}},
    ]

    state = orch.run(steps)

    # keep baseline copy
    baseline_metrics = state.get("eval_metrics")
    baseline_group_metrics = state.get("group_metrics")

    # optionally run mitigation(s)
    if use_mitigation:
        if mitigation_method == "reweight":
            # mitigation agent computes sample weights
            m_state = mitigation_agent.perform("reweight", {}, state=state)
            sample_weight = m_state.get("sample_weight")
            # retrain using weights; pass sample_weight as param into model.train
            retrain_steps = [
                {"agent": "model", "action": "train", "params": {"sample_weight": sample_weight}},
                {"agent": "model", "action": "eval", "params": {}},
                {"agent": "explain", "action": "shap", "params": {}},
            ]
            state = orch.run(retrain_steps, initial_state=state)

        elif mitigation_method == "expgrad":
            # mitigation agent returns an ExponentiatedGradient object (or similar)
            m_state = mitigation_agent.perform("expgrad", {}, state=state)
            exp_model = m_state.get("exp_model")
            # place the exp_model into state so model.eval/explain can use it
            state_with_exp = dict(state)
            state_with_exp["model"] = exp_model
            eval_steps = [
                {"agent":"model", "action":"eval", "params": {}},
                {"agent":"explain", "action":"shap", "params": {}},
            ]
            state = orch.run(eval_steps, initial_state=state_with_exp)
        else:
            raise ValueError(f"Unknown mitigation method: {mitigation_method}")

    # attach baseline snapshots for convenience
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
