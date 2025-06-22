import pandas as pd
from utils.data_utils import handle_missing_values, remove_duplicates

class CleaningAgent:
    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        explanations = []
        
        # Remove duplicates
        df, duplicate_count = remove_duplicates(df)
        explanations.append(f"Removed {duplicate_count} duplicate rows")
        
        # Handle missing values
        df, missing_explanations = handle_missing_values(df)
        explanations.extend(missing_explanations)
        
        return df, explanations