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


_model = None
_tokenizer = None

def load_model_and_tokenizer():
    """
    Load and cache the model and tokenizer.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("Loading model and tokenizer...")
        model_path = "/Users/apple/Documents/Priyesh/Pretrained-Models/all-mpnet-base-v2"
        _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _model = AutoModel.from_pretrained(model_path, local_files_only=True)
        _model.eval()
    return _tokenizer, _model


class DriftDetector:
    def __init__(self, api_key=None):
        self.report = None
        OPENAI_API_KEY = st.secrets['api_keys']["OPENAI_API_KEY"]
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide it explicitly or set it in an environment variable.")

    def _get_text_embeddings(self, texts: list) -> pd.DataFrame:
        embeddings = []
        for text in texts:
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(model="text-embedding-ada-002",
            input=text)
            embeddings.append(response.data[0].embedding)

        return pd.DataFrame(embeddings)

    def detect_tabular_drift(self, reference_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Report:
        self.report = Report(metrics=[DataDriftPreset()])
        self.report.run(reference_data=reference_data, current_data=synthetic_data)
        return self.report

    # def detect_textual_drift(self, reference_data, synthetic_data) -> Report:
    #     reference_embeddings = self._get_text_embeddings(reference_texts)
    #     synthetic_embeddings = self._get_text_embeddings(synthetic_texts)

    #     self.report = Report(metrics=[DataDriftPreset()])
    #     self.report.run(reference_data=reference_embeddings, current_data=synthetic_embeddings)
    #     return self.report

    def save_report(self, output_path: str):
        if self.report:
            self.report.save_html(output_path)


    def generate_embedding_drift_report(self, reference_data, current_data, text_column) -> str:
        """
        Generates an embedding drift report between two datasets using AutoTokenizer and AutoModel.
        
        Args:
            reference_data (pd.DataFrame): The reference dataset containing text data.
            current_data (pd.DataFrame): The current dataset containing text data.
            text_column (str): The column name containing the text data.
            
        Returns:
            str: Path to the saved HTML drift report.
        """
        print(reference_data)
        print(current_data)

        if text_column not in reference_data.columns or text_column not in current_data.columns:
            raise ValueError(f"Column '{text_column}' not found in one or both datasets.")
        
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer()
        model.eval()
        
        def get_embedding(text):
            """
            Generate embedding for a single text input.
            """
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding
        
        # Generate embeddings for reference and current data
        reference_data['embeddings'] = reference_data[text_column].apply(get_embedding)
        current_data['embeddings'] = current_data[text_column].apply(get_embedding)
        
        # Convert embeddings to DataFrame
        reference_embeddings = pd.DataFrame(reference_data['embeddings'].tolist(), columns=[f"dim_{i}" for i in range(len(reference_data['embeddings'][0]))])
        current_embeddings = pd.DataFrame(current_data['embeddings'].tolist(), columns=[f"dim_{i}" for i in range(len(current_data['embeddings'][0]))])
        
        # Generate drift report
        drift_report = Report(metrics=[
            DataDriftPreset()
        ])
        
        drift_report.run(
            reference_data=reference_embeddings,
            current_data=current_embeddings
        )
        
        report_path = "embedding_drift_report.html"
        drift_report.save_html(report_path)
        return report_path


    def generate_embedding_drift_plot(self, reference_data, current_data, text_column) -> str:
        """
        Generates an embedding drift plot (contour plot) between two datasets using AutoTokenizer and AutoModel.
        
        Args:
            reference_data (pd.DataFrame): The reference dataset containing text data.
            current_data (pd.DataFrame): The current dataset containing text data.
            text_column (str): The column name containing the text data.
            
        Returns:
            str: Path to the saved contour plot image.
        """
        if text_column not in reference_data.columns or text_column not in current_data.columns:
            raise ValueError(f"Column '{text_column}' not found in one or both datasets.")
        
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer()
        model.eval()
        
        def get_embedding(text):
            """
            Generate embedding for a single text input.
            """
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding
        
        # Generate embeddings for reference and current data
        reference_data['embeddings'] = reference_data[text_column].apply(get_embedding)
        current_data['embeddings'] = current_data[text_column].apply(get_embedding)
        
        # Concatenate embeddings
        reference_embeddings = pd.DataFrame(reference_data['embeddings'].tolist())
        current_embeddings = pd.DataFrame(current_data['embeddings'].tolist())
        
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
        
        # Plot contour plot
        plt.figure(figsize=(12, 8))
        sns.kdeplot(
            data=reduced_df,
            x='dim1', y='dim2',
            hue='dataset',
            fill=True,
            alpha=0.5,
            palette=['skyblue', 'salmon']
        )
        plt.title('Embedding Drift Contour Plot')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(title='Dataset')
        
        plot_path = 'embedding_drift_plot.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Embedding drift plot saved at: {plot_path}")
        return plot_path
    

    def display_html_report(self, report_path):
        """
        Display the HTML report in Streamlit.
        
        Args:
            report_path (str): Path to the HTML report.
        """
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=1000, scrolling=True)



