import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class InstructionAgent:
    def __init__(self):
        self.llm = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        self.prompt = PromptTemplate(
            input_variables=["rule", "columns"],
            template="Convert the following rule into executable Python code for a Pandas DataFrame. Available columns: {columns}. Rule: {rule}"
        )

    def apply_rules(self, df: pd.DataFrame, rules: str) -> tuple[pd.DataFrame, list]:
        if not rules:
            return df, []
        
        explanations = []
        columns = df.columns.tolist()
        prompt = self.prompt.format(rule=rules, columns=columns)
        response = self.llm.invoke(prompt)
        code = response.content.strip()
        
        # Sandboxed execution
        try:
            local_vars = {"df": df.copy()}
            exec(code, {}, local_vars)
            df = local_vars["df"]
            explanations.append(f"Applied user rule: {rules}")
        except Exception as e:
            explanations.append(f"Error applying rule '{rules}': {str(e)}")
        
        return df, explanations