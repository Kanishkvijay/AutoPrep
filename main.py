import streamlit as st
import pandas as pd
from workflow import AutoPrepWorkflow

st.title("AutoPrep - Automated Data Preprocessing")

# Initialize session state
if "workflow" not in st.session_state:
    st.session_state.workflow = AutoPrepWorkflow()
if "df" not in st.session_state:
    st.session_state.df = None
if "report" not in st.session_state:
    st.session_state.report = []

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(st.session_state.df.head())

# User instructions
st.subheader("Custom Preprocessing Rules")
user_rules = st.text_area("Enter rules (e.g., 'Drop column ID' or 'Fill missing Age with median')")
if st.button("Apply Custom Rules"):
    if user_rules:
        st.session_state.workflow.set_user_rules(user_rules)
        st.session_state.report.append(f"Applied user rules: {user_rules}")

# Run preprocessing
if st.button("Run Preprocessing"):
    if st.session_state.df is not None:
        result = st.session_state.workflow.run(st.session_state.df)
        st.session_state.df = result["processed_df"]
        st.session_state.report.extend(result["explanations"])
        st.write("Processed Dataset:")
        st.dataframe(st.session_state.df.head())
        st.write("Data Readiness Score:", result["readiness_score"])
        st.subheader("Preprocessing Report")
        for exp in result["explanations"]:
            st.write(exp)

# Download processed dataset
if st.session_state.df is not None:
    csv = st.session_state.df.to_csv(index=False)
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv",
    )

# Display summary report
if st.session_state.report:
    st.subheader("Summary Report")
    st.write("\n".join(st.session_state.report))