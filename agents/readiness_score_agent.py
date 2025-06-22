import pandas as pd

class ReadinessScoreAgent:
    def calculate_score(self, df: pd.DataFrame) -> float:
        score = 100.0
        
        # Missing values penalty
        missing_percentage = df.isna().mean().mean() * 100
        score -= missing_percentage * 0.5
        
        # Imbalance penalty
        for col in df.select_dtypes(include=['object', 'category']).columns:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() > 0.8:  # Highly imbalanced
                score -= 10
        
        # Feature diversity
        unique_ratio = df.nunique() / len(df)
        score -= (1 - unique_ratio.mean()) * 20
        
        return max(0, min(100, score))