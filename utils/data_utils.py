import pandas as pd
from scipy.stats import shapiro

def infer_data_types(df: pd.DataFrame) -> dict:
    return df.dtypes.to_dict()

def analyze_missing_values(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    total = len(df)
    return {col: {"count": count, "percentage": (count / total) * 100} for col, count in missing.items() if count > 0}

def detect_duplicates(df: pd.DataFrame) -> int:
    return len(df) - len(df.drop_duplicates())

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    original_len = len(df)
    df = df.drop_duplicates()
    return df, original_len - len(df)

def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    explanations = []
    for col in df.columns:
        missing_percentage = df[col].isna().mean() * 100
        if missing_percentage > 80:
            df = df.drop(columns=[col])
            explanations.append(f"Dropped column '{col}' due to >80% missing values")
        elif missing_percentage > 0:
            if df[col].dtype in ['int64', 'float64']:
                if is_normal_distribution(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    explanations.append(f"Filled missing values in '{col}' with mean")
                else:
                    df[col] = df[col].fillna(df[col].median())
                    explanations.append(f"Filled missing values in '{col}' with median")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                explanations.append(f"Filled missing values in '{col}' with mode")
    return df, explanations

def is_normal_distribution(series: pd.Series) -> bool:
    series = series.dropna()
    if len(series) < 3:
        return False
    stat, p = shapiro(series)
    return p > 0.05