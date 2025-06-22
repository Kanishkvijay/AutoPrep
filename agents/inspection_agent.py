import pandas as pd
from utils.data_utils import infer_data_types, analyze_missing_values, detect_duplicates

class InspectionAgent:
    def inspect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        explanations = []
        
        # Infer data types
        dtypes = infer_data_types(df)
        explanations.append("Inferred column data types: " + ", ".join([f"{col}: {dtype}" for col, dtype in dtypes.items()]))
        
        # Analyze missing values
        missing_info = analyze_missing_values(df)
        for col, missing in missing_info.items():
            explanations.append(f"Column '{col}' has {missing['count']} missing values ({missing['percentage']:.2f}%)")
        
        # Detect duplicates
        duplicates = detect_duplicates(df)
        explanations.append(f"Found {duplicates} duplicate rows")
        
        return df, explanations