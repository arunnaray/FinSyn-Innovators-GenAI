import os
from openai import OpenAI
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from evidently import ColumnMapping
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import (distance, mmd, model, ratio)


_model = None
_tokenizer = None

def load_model_and_tokenizer():
    """
    Load and cache the model and tokenizer.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("Loading model and tokenizer...")
        model_path = "./model"
        _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _model = AutoModel.from_pretrained(model_path, local_files_only=True)
        _model.eval()
    return _tokenizer, _model


def get_embedding(text, tokenizer, model):
    """
    Generate embedding for a single text input.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def save_report(self, output_path: str):
        if self.report:
            self.report.save_html(output_path)


def display_html_report(report_path):
        """
        Display the HTML report in Streamlit.
        
        Args:
            report_path (str): Path to the HTML report.
        """
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1000, scrolling=True)


class DriftDetector:
    def detect_tabular_drift(self, reference_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Report:
        self.report = Report(metrics=[DataDriftPreset()])
        self.report.run(reference_data=reference_data, current_data=synthetic_data)
        return self.report


    def generate_embeddings(self, reference_data, current_data, text_column) -> str:
        """
        Generates an embedding drift report between two datasets using AutoTokenizer and AutoModel.
        Uses all-mpnet-base-2 model
        
        Args:
            reference_data (pd.DataFrame): The reference dataset containing text data.
            current_data (pd.DataFrame): The current dataset containing text data.
            text_column (str): The column name containing the text data.
            
        Returns:
            return the reference and current data embeddings
        """

        if text_column not in reference_data.columns or text_column not in current_data.columns:
            raise ValueError(f"Column '{text_column}' not found in one or both datasets.")
        
        tokenizer, model = load_model_and_tokenizer()
        model.eval()
        
        # Generate embeddings for reference and current data
        reference_data['embeddings'] = reference_data[text_column].apply(get_embedding, 
                                                                         tokenizer=tokenizer, 
                                                                         model=model)
        
        current_data['embeddings'] = current_data[text_column].apply(get_embedding, 
                                                                     tokenizer=tokenizer, 
                                                                     model=model)

        return reference_data, current_data
    

    def get_textual_data_drift_preset_report(self, embedded_reference_data, embedded_current_data):
        reference_embeddings = pd.DataFrame(embedded_reference_data['embeddings'].tolist(), columns=[f"dim_{i}" for i in range(len(embedded_reference_data['embeddings'][0]))])
        current_embeddings = pd.DataFrame(embedded_current_data['embeddings'].tolist(), columns=[f"dim_{i}" for i in range(len(embedded_current_data['embeddings'][0]))])

        textual_data_drift_preset_report = Report(metrics=[
            DataDriftPreset()
        ])
        
        textual_data_drift_preset_report.run(
            reference_data=reference_embeddings,
            current_data=current_embeddings
        )
        
        textual_data_drift_preset_report_path = "./outputs/drift_reports/textual_data/textual_data_drift_preset_report.html"
        textual_data_drift_preset_report.save_html(textual_data_drift_preset_report_path)
        
        return textual_data_drift_preset_report_path


    def get_textual_data_embeddings_countour_plots(self, embedded_reference_data, embedded_current_data):
        """
        Generates 3 subplots for embedding contour plots:
        1. Reference Data
        2. Current Data
        3. Overlapping Reference and Current Data
        
        Args:
            embedded_reference_data (pd.DataFrame): DataFrame with reference embeddings.
            embedded_current_data (pd.DataFrame): DataFrame with current embeddings.
            
        Returns:
            str: Path to the saved subplot image.
        """
        # Prepare Embeddings DataFrames
        reference_embeddings = pd.DataFrame(embedded_reference_data['embeddings'].tolist())
        current_embeddings = pd.DataFrame(embedded_current_data['embeddings'].tolist())
        
        reference_embeddings['dataset'] = 'Reference'
        current_embeddings['dataset'] = 'Current'
        
        combined_embeddings = pd.concat([reference_embeddings, current_embeddings], ignore_index=True)
        labels = combined_embeddings['dataset']
        combined_embeddings = combined_embeddings.drop(columns=['dataset'])
        
        # Dimensionality Reduction using t-SNE
        print("Performing dimensionality reduction using t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(combined_embeddings)
        
        reduced_df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2'])
        reduced_df['dataset'] = labels.values
        
        # Create Subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Reference Data
        sns.kdeplot(
            data=reduced_df[reduced_df['dataset'] == 'Reference'],
            x='dim1', y='dim2',
            fill=True,
            alpha=0.5,
            color='skyblue',
            ax=axs[0]
        )
        axs[0].set_title('Reference Data Embedding Contour')
        axs[0].set_xlabel('Dimension 1')
        axs[0].set_ylabel('Dimension 2')
        
        # Plot 2: Current Data
        sns.kdeplot(
            data=reduced_df[reduced_df['dataset'] == 'Current'],
            x='dim1', y='dim2',
            fill=True,
            alpha=0.5,
            color='salmon',
            ax=axs[1]
        )
        axs[1].set_title('Current Data Embedding Contour')
        axs[1].set_xlabel('Dimension 1')
        axs[1].set_ylabel('Dimension 2')
        
        # Plot 3: Overlapping Contour
        sns.kdeplot(
            data=reduced_df,
            x='dim1', y='dim2',
            hue='dataset',
            fill=True,
            alpha=0.5,
            palette=['skyblue', 'salmon'],
            ax=axs[2]
        )
        axs[2].set_title('Overlap: Reference & Current Data')
        axs[2].set_xlabel('Dimension 1')
        axs[2].set_ylabel('Dimension 2')
        axs[2].legend(title='Dataset')
        
        # Adjust Layout and Save Plot
        plt.tight_layout()
        textual_data_embeddings_contour_plots_path = './outputs/drift_reports/textual_data/textual_data_embeddings_contour_plots.png'
        plt.savefig(textual_data_embeddings_contour_plots_path)
        plt.close()
        
        print(f"Embedding contour plots saved at: {textual_data_embeddings_contour_plots_path}")
        return textual_data_embeddings_contour_plots_path
    

    def get_embeddings_drift_reports(self, embedded_reference_data, embedded_current_data):
        ref_embeddings = embedded_reference_data['embeddings'].to_list()
        curr_embeddings = embedded_current_data['embeddings'].to_list() 

        ref_embeddings_df = pd.DataFrame(ref_embeddings)
        ref_embeddings_df.columns = ['col_' + str(x) for x in ref_embeddings_df.columns]

        curr_embeddings_df = pd.DataFrame(curr_embeddings)
        curr_embeddings_df.columns = ['col_' + str(x) for x in curr_embeddings_df.columns]

        column_mapping = ColumnMapping(embeddings={'Synthetic Data Generation' : ref_embeddings_df.columns})

        embedding_drif_mmd_report = Report(metrics= [
            EmbeddingsDriftMetric('Synthetic Data Generation',
                                drift_method=mmd(
                                        threshold = 0.5,
                                        bootstrap = False,
                                        quantile_probability = 0.5,
                                        pca_components=None
                                ))
        ])

        embedding_drif_mmd_report.run(reference_data=ref_embeddings_df,
                                    current_data=curr_embeddings_df,
                                    column_mapping=column_mapping)
        
        textual_embeddings_drift_mmd_report_path = "./outputs/drift_reports/textual_data/textual_embeddings_drift_mmd_report.html"
        embedding_drif_mmd_report.save_html(textual_embeddings_drift_mmd_report_path)
    
        return textual_embeddings_drift_mmd_report_path


    def textual_data_drift_reports(self, reference_data, current_data, text_column):
        embedded_reference_data, embedded_current_data = self.generate_embeddings(reference_data,
                                                                                  current_data,
                                                                                  text_column)
        
        textual_data_drift_preset_report_path = self.get_textual_data_drift_preset_report(embedded_reference_data,
                                                                                          embedded_current_data)
        textual_data_embeddings_countour_plots_path = self.get_textual_data_embeddings_countour_plots(embedded_reference_data,
                                                                                                      embedded_current_data)
        textual_embeddings_drift_mmd_report_path = self.get_embeddings_drift_reports(embedded_reference_data,
                                                                                    embedded_current_data)
        
        return textual_data_drift_preset_report_path, \
                textual_data_embeddings_countour_plots_path, \
                textual_embeddings_drift_mmd_report_path
    

    def show_info(self):
        st.markdown(f"<h5 style='text-align: center;'> Graph Interpretations </h5>", unsafe_allow_html=True)
        st.success('''The graphs in the report compare the Reference Distribution (historical or baseline data) with the Current Distribution (new incoming data) for each feature or column. Each pair of histograms visualizes how data is distributed across different bins or categories. When the distributions are visually aligned, it suggests stability in the feature’s behavior over time. However, noticeable differences—such as shifts in peaks, spread, or gaps—indicate potential data drift. The statistical tests (e.g., K-S test for numerical data and Chi-Square test for categorical data) provide quantitative validation of these visual observations. The drift score and p-values further indicate whether the drift is statistically significant.''')
        st.warning('''
                    Key Observations to look For:
                    •	Alignment of Bars: If the bars in the reference and current distributions overlap closely, drift is unlikely.
                    •	Shift in Peaks: Changes in the central tendency or concentration of data suggest drift.
                    •	Spread and Shape: Variations in data spread or the overall shape of distributions signal potential drift.
                    •	Statistical Significance:
                    •	High p-value (>0.05): No significant drift detected.
                    •	Low p-value (<0.05): Significant drift detected.
                    •	Drift Score: A higher drift score indicates greater deviation from the reference distribution.
        ''')
        st.markdown(f"<h5 style='text-align: center;'> Statistical Tests Used </h5>", unsafe_allow_html=True)
        st.error('''
                Kolmogorov-Smirnov (K-S) Test:
                    •	The K-S test compares the cumulative distributions of two datasets (reference vs. current).
                    •	It measures the maximum difference between their cumulative distribution functions (CDFs).
                    •	A high p-value (e.g., > 0.05) suggests no significant drift, while a low p-value indicates data drift.
                    •	In the plots, significant visual shifts between the two distributions suggest drift, aligning with the statistical results.

                Chi-Square Test:
                    •	The Chi-Square test compares the frequency distributions of categorical data or discretized numeric data.
                    •	It evaluates whether the observed frequencies in the current dataset deviate significantly from the expected frequencies (reference dataset).
                    •	A low p-value (e.g., < 0.05) suggests significant drift in categorical distributions.
                    •	Large visual differences in the histogram bars between reference and current indicate drift.

                Population Stability Index (PSI):
                    •	PSI quantifies the shift in distributions between reference and current datasets.
                    •	A PSI score < 0.1 indicates no significant drift, 0.1–0.25 indicates moderate drift, and > 0.25 suggests significant drift.
                    •	While PSI isn’t explicitly shown in this report, it is often used alongside these methods to provide an additional measure of drift.
        ''')

        synthetic_data_csv = st.session_state.synthetic_data.to_csv(index=False)  
        st.download_button(
            label="Download Synthetic Data",
            data=synthetic_data_csv,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )
        

        



