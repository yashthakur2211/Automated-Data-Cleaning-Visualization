from diffprivlib.mechanisms import Laplace

def apply_privacy_preservation(df):
    # Apply Laplace noise to a sensitive column for privacy preservation
    mech = Laplace(epsilon=1, sensitivity=1)
    df['sensitive_column'] = df['sensitive_column'].apply(lambda x: mech.randomise(x))
    return df
