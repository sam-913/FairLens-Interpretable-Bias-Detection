# pmas/orchestrator.py
from typing import Dict, Any, Optional
import logging

class Orchestrator:
    def __init__(self):
        self.agents = {}
        self.logger = logging.getLogger("pmas.orch")

    def register(self, name: str, agent):
        self.agents[name] = agent

    def run(self, workflow: list, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a workflow (list of steps). Optionally provide an `initial_state` dict
        so this run continues from previous state (useful for mitigation/retrain).
        Each step: {"agent": "<name>", "action": "<action>", "params": {...}}
        """
        # start from a copy of initial_state to avoid mutating caller's dict unexpectedly
        state = dict(initial_state) if initial_state is not None else {}
        for step in workflow:
            agent_name = step["agent"]
            action = step["action"]
            params = step.get("params", {})
            agent = self.agents.get(agent_name)
            if agent is None:
                raise RuntimeError(f"Agent '{agent_name}' not registered.")
            self.logger.info("Running %s.%s", agent_name, action)
            out = agent.perform(action, params, state=state)
            if isinstance(out, dict):
                state.update(out)
        return state
