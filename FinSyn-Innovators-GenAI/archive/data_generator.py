from openai import OpenAI
import pandas as pd

class SyntheticDataGenerator:
    def __init__(self, api_key):
        self.api_key = api_key


    def generate_tabular_data(self, reference_data: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        prompt = f"Generate {num_rows} rows of synthetic data based on the following schema:\n{reference_data.head().to_string(index=False)}"
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a data generation assistant."},
            {"role": "user", "content": prompt}
        ])
        synthetic_data_text = response.choices[0].message.content
        synthetic_data = pd.read_csv(pd.compat.StringIO(synthetic_data_text))
        return synthetic_data

    def generate_textual_data(self, reference_text: str, num_samples: int) -> list:
        prompt = f"Generate {num_samples} synthetic samples based on the following text:\n{reference_text}"
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a text generation assistant."},
            {"role": "user", "content": prompt}
        ])
        synthetic_texts = response.choices[0].message.content.split("\n")
        return synthetic_texts
