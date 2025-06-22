import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.data_utils import is_normal_distribution

class ScalingAgent:
    def scale(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        explanations = []
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_cols:
            if is_normal_distribution(df[col]):
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
                explanations.append(f"Applied StandardScaler to '{col}' (normal distribution)")
            else:
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
                explanations.append(f"Applied MinMaxScaler to '{col}' (non-normal distribution)")
        
        return df, explanations