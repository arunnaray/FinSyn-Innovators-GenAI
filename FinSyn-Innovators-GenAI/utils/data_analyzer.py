import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from scipy.stats import norm
import numpy as np
import streamlit as st
import time


class DataAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm_client = OpenAI(api_key=self.api_key)
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.
        """
        return data.describe(include='all').transpose()
    
    
    def generate_column_plot(self, data: pd.DataFrame, column: str) -> BytesIO:
        """
        Generate distribution and box plot for a given column.
        """
        buffer = BytesIO()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution Plot
        sns.histplot(data[column].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution Plot for {column}')
        
        # Box Plot
        sns.boxplot(y=data[column].dropna(), ax=axes[1])
        axes[1].set_title(f'Box Plot for {column}')
        
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return buffer


    def generate_column_plot_plotly(self, data: pd.DataFrame, column: str) -> dict:
        """
        Generate multiple interactive plots for a given column using Plotly.
        The plots are generated only for numerical columns.
        """
        plots = {}
        
        # Check if the column is numerical
        if not np.issubdtype(data[column].dtype, np.number):
            # Skip generating plots for non-numeric columns
            return plots
        
        # Gaussian Distribution Plot (for Numerical Columns)
        mean = data[column].mean()
        std = data[column].std()
        x = np.linspace(data[column].min(), data[column].max(), 100)
        y = norm.pdf(x, mean, std)
        
        fig_gaussian = go.Figure()
        fig_gaussian.add_trace(go.Histogram(
            x=data[column],
            histnorm='probability density',
            name='Data Distribution',
            opacity=0.7
        ))
        fig_gaussian.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Gaussian Fit',
            line=dict(color='red')
        ))
        fig_gaussian.update_layout(
            title=f"Gaussian Distribution Plot for {column}",
            xaxis_title=column,
            yaxis_title="Density"
        )
        plots['gaussian_distribution_plot'] = fig_gaussian

        # Distribution Plot
        fig_dist = px.histogram(data, x=column, marginal="box", nbins=30, title=f"Distribution Plot for {column}")
        plots['distribution_plot'] = fig_dist
        
        # Box Plot
        fig_box = px.box(data, y=column, title=f"Box Plot for {column}")
        plots['box_plot'] = fig_box
        
        # Outlier Detection Plot
        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Box(y=data[column], boxpoints='all', name='Outliers'))
        fig_outliers.update_layout(title=f"Outlier Detection for {column}")
        plots['outlier_plot'] = fig_outliers

        return plots



    def generate_column_insight(self, column_name: str, stats: pd.Series) -> str:
        """
        Generate insights for a column using LLM.
        """
        with st.spinner(f"Gererating statistical insights for {column_name}..."):
            progress = st.progress(0)
            for i in range(100):  
                time.sleep(0.02) 
                progress.progress(i + 1)
        prompt = f"""
        Analyze the following statistical summary for the column '{column_name}':
        {stats.to_string()}
        
        Provide a concise summary and key insights based on this information.
        Limit the summary with in 50 words.
        """
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    

    def show_plots_and_insights(self, dataset):
        # columns = ['ID', 'Salary', 'Years of Experience', 
        #            'Last Performance Rating', 'Laptop Assigned']
        #columns = ['Salary', 'Department']
        columns = dataset.columns.tolist()
        columns = columns[0:5]
        #num_columns = [i for i in columns if np.issubdtype(dataset[i].dtype, np.number)]
        #print(num_columns)
        summary_stats = self.generate_summary_statistics(dataset)
        st.dataframe(summary_stats)
        for column in columns:
            st.write(f"### Column: {column}")
            column_plots = self.generate_column_plot_plotly(dataset, column)
            if column_plots:
                for plot_name, fig in column_plots.items():
                    st.write(f"**{plot_name.replace('_', ' ').title()}**")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                pass
                #st.write(f"No plots available for column: {column} (non-numeric data).")

            column_stats = summary_stats.loc[column]
            column_insight = self.generate_column_insight(column, column_stats)
            st.write("**Insights:**")
            st.info(column_insight)
