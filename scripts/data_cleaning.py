import pandas as pd

def clean_data(df):
    # Initialize cleaning stats dictionary
    cleaning_stats = {
        'Missing Values Filled': 0,
        'Outliers Removed': 0  # Placeholder, to be modified if handling outliers
    }

    # Select numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    non_numeric_cols = df.select_dtypes(exclude='number').columns

    # Count missing values before cleaning
    missing_before = df.isnull().sum().sum()

    # Apply mean imputation to numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill missing values in non-numeric columns with 'Unknown'
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')

    # Count missing values after cleaning
    missing_after = df.isnull().sum().sum()

    # Update cleaning stats
    cleaning_stats['Missing Values Filled'] = missing_before - missing_after

    # Return both the cleaned dataframe and the cleaning statistics
    return df, cleaning_stats
