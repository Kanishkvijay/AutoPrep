from utils.explanation_utils import format_explanation

class ExplainabilityAgent:
    def generate_explanations(self, explanations: list) -> list:
        return [format_explanation(exp) for exp in explanations]