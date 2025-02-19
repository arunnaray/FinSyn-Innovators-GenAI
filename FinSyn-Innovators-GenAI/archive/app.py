import streamlit as st
import pandas as pd
from utils.data_generator import SyntheticDataGenerator
from utils.drift_detector import DriftDetector

# API Key Configuration
OPENAI_API_KEY = st.secrets['api_keys']["OPENAI_API_KEY"]

# Initialize Classes
data_gen = SyntheticDataGenerator(api_key=OPENAI_API_KEY)
drift_detector = DriftDetector()

# Streamlit UI
st.title("Synthetic Data Generation App")

# Tabs for User Flow
tab1, tab2 = st.tabs(["Reference Data Input", "Metadata Input (Coming Soon)"])

# Scenario 1: Reference Data Input
with tab1:
    st.header("Generate Synthetic Data from Reference Data")
    data_type = st.selectbox("Select Data Type", ["Tabular", "Textual"])

    if data_type == "Tabular":
        uploaded_file = st.file_uploader("Upload Reference CSV", type=["csv"])

        if uploaded_file:
            reference_data = pd.read_csv(uploaded_file)
            st.write("Reference Data Preview:", reference_data.head())

            num_rows = st.number_input("Number of Synthetic Rows", min_value=10, value=100)
            if st.button("Generate Synthetic Tabular Data"):
                synthetic_data = data_gen.generate_tabular_data(reference_data, num_rows)
                st.write("Synthetic Data Preview:", synthetic_data.head())

                if st.button("Run Drift Detection"):
                    drift_report = drift_detector.detect_tabular_drift(reference_data, synthetic_data)
                    drift_report.save_html("tabular_drift_report.html")
                    st.success("Drift Detection Report Generated! Download below.")
                    st.download_button("Download Report", "tabular_drift_report.html")

    elif data_type == "Textual":
        uploaded_file = st.file_uploader("Upload Reference CSV for Text Data", type=["csv"])
        text_column = st.text_input("Enter the Text Column Name", placeholder="e.g., text")

        if uploaded_file and text_column:
            reference_data = pd.read_csv(uploaded_file)
            st.dataframe(reference_data)

            if text_column not in reference_data.columns:
                st.error(f"Column '{text_column}' not found in the uploaded CSV.")
            else:
                reference_texts = reference_data[text_column].dropna().tolist()
                st.write("Sample Reference Texts:", reference_texts[:5])

                num_samples = st.number_input("Number of Synthetic Text Samples", min_value=1, value=5)
                if st.button("Generate Synthetic Textual Data"):
                    synthetic_texts = data_gen.generate_textual_data("\n".join(reference_texts), num_samples)
                    st.write("Synthetic Textual Data:", synthetic_texts)

                    if st.button("Run Drift Detection"):
                        drift_report = drift_detector.detect_textual_drift(reference_texts, synthetic_texts)
                        drift_report.save_html("textual_drift_report.html")
                        st.success("Drift Detection Report Generated! Download below.")
                        st.download_button("Download Report", "textual_drift_report.html")

# Scenario 2: Placeholder for Metadata-Based Generation
with tab2:
    st.header("Generate Synthetic Data from Metadata (Coming Soon)")
    st.info("This feature will be implemented in the next phase.")