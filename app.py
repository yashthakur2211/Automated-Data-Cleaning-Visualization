import streamlit as st
import pandas as pd
import tempfile
import spacy
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("	https://cdn.pixabay.com/photo/2013/08/09/05/54/layer-170971_1280.jpg");
        background-size: cover;
        background-position: center;
        filter: blur(0px); /* Adjust the blur level here */
    }
    
    /* Add a white overlay on top of the background to improve text visibility */
    [data-testid="stAppViewContainer"]:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.3); /* Semi-transparent white overlay */
        z-index: -1; /* Keep it behind the text */
    }

    /* Light Mode Text Color */
    @media (prefers-color-scheme: light) {
        body {
            color: black;
        }
    }

    /* Dark Mode Text Color */
    @media (prefers-color-scheme: dark) {
        body {
            color: white;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a feature:", ["Data Cleaning", "Data Visualization"])

# Data Cleaning Function (as defined earlier)
def clean_data(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        df[col].fillna(df[col].mean(), inplace=True)
    
    cleaning_stats = {"Missing Values Filled": df[numeric_columns].isnull().sum().sum()}
    return df, cleaning_stats

# Sensitive Data Detection Function (as defined earlier)
def detect_sensitive_data(df):
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        sensitive_columns = []
        for text in df[col].fillna(''):
            doc = nlp(text)
            sensitive_info = [(ent.text, ent.label_) for ent in doc.ents]
            sensitive_columns.append(sensitive_info)
        df[f'sensitive_info_{col}'] = sensitive_columns
    return df

# Main logic for different pages
if options == "Data Cleaning":
    st.title("Automated Data Cleaning and Sensitive Data Detection Tool")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            st.warning("Could not decode file using UTF-8, trying ISO-8859-1")
            data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        st.write("Original Dataset", data)

        # Clean the data and capture stats
        cleaned_data, cleaning_stats = clean_data(data)
        st.write("Cleaned Dataset", cleaned_data)

        # Display cleaning statistics
        st.write(f"Missing Values Filled: {cleaning_stats['Missing Values Filled']}")

        # Detect sensitive data
        sensitive_data_detected = detect_sensitive_data(cleaned_data)
        st.write("Dataset with Sensitive Data Detection", sensitive_data_detected)

        # Save processed data to a temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            sensitive_data_detected.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        # Download button
        st.download_button(
            label="Download Processed Data",
            data=open(tmp_file_path, 'rb').read(),
            file_name='processed_data.csv',
            mime='text/csv'
        )

elif options == "Data Visualization":
    st.title("Data Visualization of Cleaned Data")
    
    # Assuming cleaned_data is available from the previous data cleaning
    # In case it's not, prompt the user to upload the data again.
    uploaded_file = st.file_uploader("Upload Cleaned CSV file", type="csv")
    if uploaded_file is not None:
        try:
            cleaned_data = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            st.warning("Could not decode file using UTF-8, trying ISO-8859-1")
            cleaned_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        st.write("Cleaned Dataset", cleaned_data)
        
        # Select columns for visualization
        numeric_columns = cleaned_data.select_dtypes(include=['number']).columns
        selected_col = st.selectbox("Select a column to visualize:", numeric_columns)

        # Display various plots
        if st.checkbox("Show Histogram"):
            fig = px.histogram(cleaned_data, x=selected_col)
            st.plotly_chart(fig)

        if st.checkbox("Show Box Plot"):
            fig = px.box(cleaned_data, y=selected_col)
            st.plotly_chart(fig)

        if st.checkbox("Show Correlation Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(cleaned_data[numeric_columns].corr(), annot=True, ax=ax)
            st.pyplot(fig)
