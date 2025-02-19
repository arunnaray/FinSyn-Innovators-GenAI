from openai import OpenAI
import json
import streamlit as st

class DataGenerationUsingMetaInfo:

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def parse_llm_schema(self, schema_str: str) -> dict:
            """
            Parses an LLM-generated JSON schema string and returns a dictionary
            where keys are field names and values are their properties.
            
            Args:
                schema_str (str): The JSON schema string returned by the LLM.

            Returns:
                dict: Dictionary with field names as keys and their properties as values.
            """
            try:
                # Parse the JSON string into a Python dictionary
                schema = json.loads(schema_str)
            except json.JSONDecodeError as e:
                print("❌ Failed to parse JSON:", e)
                return {}

            # Extract properties
            properties = schema.get('properties', {})
            result = {}

            for field, details in properties.items():
                result[field] = {
                    "type": details.get('type', 'Unknown'),
                    "description": details.get('description', 'No Description'),
                    "format": details.get('format', 'N/A') if 'format' in details else None,
                    "required": field in schema.get('required', [])
                }
            return result
        

    def get_metadata_from_llm(self, user_prompt: str) -> dict:
        """
        Sends the user prompt to the LLM and returns a JSON schema suggestion.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant designed to create JSON schemas for synthetic data generation. Always respond in valid JSON format. Given a user description, suggest a JSON schema with column names, data types, and constraints."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        try:
            metadata_schema_str = response.choices[0].message.content
            schema = self.parse_llm_schema(metadata_schema_str)
            return schema

        except json.JSONDecodeError as e:
            st.error("Failed to parse LLM response as JSON. Please refine your input.")
            st.write("🛠️ **Debug Info:** JSON Decode Error")
            st.write(str(e))
            return {}

        except (KeyError, AttributeError) as e:
            st.error(f"Unexpected response format: {e}")
            return {}


    def generate_synthetic_data_llm(self, schema, field_ranges, num_records):
        """
        Send schema, field ranges, and number of records to LLM for data generation.
        Here, OpenAI GPT is assumed, replace with your LLM API.
        """

        # Construct the prompt
        prompt = f"""
        Given the following schema and field constraints, generate {num_records} records of synthetic data.
        The data should be realistic and varied, with randomized values for each field based on the schema.
        Ensure to generate different names, roles, designations, employee IDs, salaries, and departments.
        Also ensure that the response you give should only contain the synthetic data. No unnecesary text.

        Schema: {json.dumps(schema, indent=4)}
        Field Ranges: {json.dumps(field_ranges, indent=4)}
        Please generate the data in CSV format with one record per row. Each record should have unique values for fields like employee_id, name, role, designation, salary, and department.
        """

        # Ensure you have the correct model, e.g., "gpt-3.5-turbo"
        model = "gpt-3.5-turbo"

        # Create the messages for the chat API
        messages = [{"role": "system", "content": "You are a helpful assistant for generating synthetic data. Note: when you generate the data, do not add duplicate values in the data. augment the data with different values for each record."}]
        messages.append({"role": "user", "content": prompt})

        # Make the API call to OpenAI for chat completion
        response = self.client.chat.completions.create(
            model=model,   # Specify the model
            messages=messages,  # Pass the messages
            max_tokens=2000,
            n=1,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()