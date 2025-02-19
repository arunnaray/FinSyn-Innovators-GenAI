from openai import OpenAI
import pandas as pd
import json
import streamlit as st
import csv
from io import StringIO

class SyntheticDataGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
    

    def generate_tabular_data(self, reference_data: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        schema_description = []
        for column in reference_data.columns:
            dtype = reference_data[column].dtype
            if dtype in ['int64', 'float64']:
                summary = f"mean: {reference_data[column].mean():.2f}, std: {reference_data[column].std():.2f}, min: {reference_data[column].min()}, max: {reference_data[column].max()}"
            elif dtype == 'object':
                unique_values = reference_data[column].nunique()
                sample_values = reference_data[column].dropna().unique()[:3]
                summary = f"{unique_values} unique values, e.g., {list(sample_values)}"
            else:
                summary = "Non-numeric data"
            schema_description.append(f"{column} ({dtype}): {summary}")
        
        schema_summary = "\n".join(schema_description)

        # Create the prompt
        prompt = f"""
        Generate {num_rows} rows of synthetic data in CSV format based on the following schema:
        {schema_summary}

        The generated data should align with the described schema and statistical properties.
        Provide the output in CSV format enclosed by START_CSV and END_CSV placeholders.
        """
        
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data generation assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract synthetic data from the response
        synthetic_data_text = response.choices[0].message.content
        start_index = synthetic_data_text.find("START_CSV") + len("START_CSV")
        end_index = synthetic_data_text.find("END_CSV")
        csv_data = synthetic_data_text[start_index:end_index].strip()
        synthetic_data = pd.read_csv(StringIO(csv_data))
        return synthetic_data



    def generate_textual_data(self, reference_text: str, num_samples: int) -> list:
        prompt = f"Generate {num_samples} synthetic samples based on the following text:\n{reference_text}"
        response = self.client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a text generation assistant."},
            {"role": "user", "content": prompt}
        ])
        synthetic_texts = response.choices[0].message.content.split("\n")
        return synthetic_texts
    
    

    def generate_textual_data_with_schema(self, reference_data: pd.DataFrame, text_column: str, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic textual data while preserving the schema of the reference dataset.
        """
        import csv
        from io import StringIO
        
        # Extract a few sample rows for context
        sample_rows = reference_data.head(5).to_dict(orient='records')
        schema = list(reference_data.columns)
        
        # Build the prompt
        prompt = f"""
        Generate {num_samples} rows of synthetic data strictly following the schema:
        Columns: {schema}
        
        - Focus on generating synthetic text for the column: '{text_column}'.
        - For other columns ('Category' and 'Product'), provide contextually relevant synthetic values.
        - Ensure the output matches the schema exactly: {schema}
        - The output must be in CSV format with the header row: {','.join(schema)}
        
        Here are some example rows from the dataset:
        {json.dumps(sample_rows, indent=2)}
        
        Example Output:
        {','.join(schema)}
        Synthetic incident text 1,Category Example,Product Example
        Synthetic incident text 2,Another Category,Another Product
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant generating synthetic data that strictly follows a given schema."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the response
        response_text = response.choices[0].message.content.strip()
        
        # Check if the response starts with the correct schema
        reader = csv.reader(StringIO(response_text))
        headers = next(reader)
        headers = [h.strip() for h in headers]
        
        if headers != schema:
            raise ValueError(
                f"Schema mismatch!\nExpected: {schema}\nGot: {headers}\nResponse:\n{response_text}"
            )
        
        # Load the CSV data into a DataFrame
        synthetic_data = pd.DataFrame(reader, columns=headers)
        return synthetic_data






        

        
