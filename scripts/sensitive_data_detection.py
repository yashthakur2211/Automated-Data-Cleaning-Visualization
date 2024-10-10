import pandas as pd
import spacy

# Load spaCy's small English model (download it using: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def detect_sensitive_data(df):
    # Automatically detect text columns (object type in pandas)
    text_columns = df.select_dtypes(include=['object']).columns

    if len(text_columns) == 0:
        # If no text columns are found, return a warning or error message
        print("No text columns found for sensitive data detection.")
        return df
    
    # Loop over all detected text columns and process them
    for col in text_columns:
        sensitive_columns = []
        for text in df[col].fillna(''):  # Handle missing values by filling with empty string
            doc = nlp(text)
            sensitive_info = [(ent.text, ent.label_) for ent in doc.ents]
            sensitive_columns.append(sensitive_info)
        
        # Add a new column with detected sensitive information for each text column
        df[f'sensitive_info_{col}'] = sensitive_columns
    
    return df
