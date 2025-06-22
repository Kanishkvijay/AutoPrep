from langgraph.graph import StateGraph, END
from typing import Dict, Any
import pandas as pd
from agents.inspection_agent import InspectionAgent
from agents.cleaning_agent import CleaningAgent
from agents.encoding_agent import EncodingAgent
from agents.scaling_agent import ScalingAgent
from agents.instruction_agent import InstructionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.readiness_score_agent import ReadinessScoreAgent

class AutoPrepState(Dict[str, Any]):
    df: pd.DataFrame
    explanations: list
    user_rules: str
    readiness_score: float

class AutoPrepWorkflow:
    def __init__(self):
        self.graph = self._build_graph()
        self.explanations = []

    def _build_graph(self):
        graph = StateGraph(AutoPrepState)
        
        graph.add_node("inspection", self._inspection_node)
        graph.add_node("instruction", self._instruction_node)
        graph.add_node("cleaning", self._cleaning_node)
        graph.add_node("encoding", self._encoding_node)
        graph.add_node("scaling", self._scaling_node)
        graph.add_node("explainability", self._explainability_node)
        graph.add_node("readiness", self._readiness_node)

        graph.set_entry_point("inspection")
        graph.add_edge("inspection", "instruction")
        graph.add_edge("instruction", "cleaning")
        graph.add_edge("cleaning", "encoding")
        graph.add_edge("encoding", "scaling")
        graph.add_edge("scaling", "explainability")
        graph.add_edge("explainability", "readiness")
        graph.add_edge("readiness", END)

        return graph.compile()

    def _inspection_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = InspectionAgent()
        state["df"], inspection_explanations = agent.inspect(state["df"])
        state["explanations"].extend(inspection_explanations)
        return state

    def _instruction_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = InstructionAgent()
        state["df"], instruction_explanations = agent.apply_rules(state["df"], state["user_rules"])
        state["explanations"].extend(instruction_explanations)
        return state

    def _cleaning_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = CleaningAgent()
        state["df"], cleaning_explanations = agent.clean(state["df"])
        state["explanations"].extend(cleaning_explanations)
        return state

    def _encoding_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = EncodingAgent()
        state["df"], encoding_explanations = agent.encode(state["df"])
        state["explanations"].extend(encoding_explanations)
        return state

    def _scaling_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = ScalingAgent()
        state["df"], scaling_explanations = agent.scale(state["df"])
        state["explanations"].extend(scaling_explanations)
        return state

    def _explainability_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = ExplainabilityAgent()
        state["explanations"] = agent.generate_explanations(state["explanations"])
        return state

    def _readiness_node(self, state: AutoPrepState) -> AutoPrepState:
        agent = ReadinessScoreAgent()
        state["readiness_score"] = agent.calculate_score(state["df"])
        state["explanations"].append(f"Data Readiness Score: {state['readiness_score']}")
        return state

    def set_user_rules(self, rules: str):
        self.graph.state["user_rules"] = rules

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        state = AutoPrepState(df=df, explanations=[], user_rules="", readiness_score=0.0)
        result = self.graph.invoke(state)
        return {
            "processed_df": result["df"],
            "explanations": result["explanations"],
            "readiness_score": result["readiness_score"]
        }