import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class EncodingAgent:
    def encode(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        explanations = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals == 2:  # Binary categorical
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                explanations.append(f"Applied Label Encoding to '{col}' (binary categorical)")
            elif unique_vals <= 10:  # Nominal categorical with reasonable cardinality
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                encoded = pd.DataFrame(ohe.fit_transform(df[[col]]), columns=ohe.get_feature_names_out([col]))
                df = pd.concat([df.drop(columns=[col]), encoded], axis=1)
                explanations.append(f"Applied One-Hot Encoding to '{col}' (nominal categorical)")
        
        return df, explanations