import streamlit as st
import pandas as pd
from datetime import datetime
import re
import os
from fuzzywuzzy import process, fuzz  # For fuzzy string matching

# Define weights based on the updated formula
weights = {
    'w1': 0.3,   # Relevance
    'w2': 0.25,  # NHI (Normalized H-Index, scaled 0-10)
    'w3': 0.2,   # Recency
    'w4': 0.15,  # Study Design
    'w5': 0.1    # p-value
}

# Function to calculate relevance based on keyword matching
def calculate_relevance(text, keyword_weights):
    if pd.isna(text):  # Handle NaN values
        return 0
    relevance_score = 0
    for keyword, weight in keyword_weights.items():
        if keyword.lower() in str(text).lower():  # Ensure text is treated as a string
            if weight > relevance_score:
                relevance_score = weight
    return relevance_score

# Function to calculate recency score based on publication year
def calculate_recency(pub_year):
    if pd.isna(pub_year):  # Handle NaN values
        return 3  # Default score for missing publication year
    current_year = datetime.now().year
    years_ago = current_year - int(pub_year)  # Ensure pub_year is treated as an integer

    if years_ago == 0:
        return 10  # Published current year
    elif years_ago == 1:
        return 8   # Published last year
    elif years_ago == 2:
        return 7   # Published 2 years ago
    elif years_ago == 3:
        return 6   # Published 3 years ago
    elif years_ago == 4:
        return 5   # Published 4 years ago
    else:
        return 3   # Published >5 years ago

# Function to extract p-value from text and assign a score
def extract_p_value_score(text):
    if pd.isna(text):  # Handle NaN values
        return 5  # Default score if no text is found

    # Regular expression to find p-values in the text
    p_value_pattern = r'p\s*[<=>]\s*([0-9]*\.?[0-9]+)'
    matches = re.findall(p_value_pattern, str(text), re.IGNORECASE)  # Ensure text is treated as a string

    if not matches:
        return 3  # Default score if no p-value is found

    # Extract the smallest p-value (most significant)
    p_values = [float(match) for match in matches]
    min_p_value = min(p_values)

    # Assign scores based on p-value ranges
    if min_p_value < 0.001:
        return 10
    elif min_p_value < 0.01:
        return 9
    elif min_p_value < 0.025:
        return 8
    elif min_p_value < 0.04:
        return 7
    elif min_p_value < 0.05:
        return 6
    else:
        return 3

# Function to calculate study design score based on title and abstract
def calculate_study_design(title, abstract):
    """
    Calculate study design score based on title and abstract using fuzzy matching.
    """
    if pd.isna(title) or pd.isna(abstract):  # Handle NaN values
        return 0  # Default score if no title or abstract is found

    # Combine title and abstract into a single text
    text = f"{title} {abstract}".lower()

    # Define study design keywords and their corresponding scores
    study_design_keywords = {
        "randomized controlled trial": 10,
        "rct": 10,
        "randomized trial": 10,
        "randomized clinical trial": 10,
        "randomised trial": 10,
        "randomised controlled trial": 10,
        "systematic review": 9,
        "meta-analysis": 9,
        "multicenter": 8,
        "multicentre": 8,
        "multi-center": 8,
        "multi institutional": 8,
        "prospective cohort study": 7,
        "cohort study": 7,
        "cross-sectional study": 6,
        "case-control study": 5,
        "case series": 4,
        "case report": 4,
        "expert opinion": 2,
        "editorial": 2,
        "animal study": 1,
        "in vitro study": 1,
        "anecdotal": 0,
    }

    # Initialize the highest score
    highest_score = 3  # Default score if no study design is found

    # Check each keyword using fuzzy matching
    for keyword, score in study_design_keywords.items():
        if fuzz.partial_ratio(keyword, text) >= 80:  # Adjust threshold as needed
            if score > highest_score:
                highest_score = score  # Keep the highest score

    return highest_score

# Function to perform fuzzy matching for journal names
def fuzzy_match_journal(journal, journal_list, threshold=90):
    """
    Perform fuzzy matching to find the best match for a journal name.
    :param journal: The journal name to match.
    :param journal_list: List of known journal names.
    :param threshold: Minimum match score (0-100) to consider a match.
    :return: The best match if above the threshold, otherwise None.
    """
    if pd.isna(journal):  # Handle NaN values
        return None

    # Use fuzzy matching to find the best match
    match, score = process.extractOne(journal, journal_list)

    # Only return the match if the score is above the threshold
    if score >= threshold:
        return match
    else:
        return None  # No valid match found

# Streamlit app
def main():
    st.title("Document Evaluation Tool")
    st.write("Upload your document for evaluation.")

    # Load the normalized h-index file from the resources folder
    h_index_folder = "resources"  # Folder where the h-index file is stored
    h_index_file = "normalized_h_index.xlsx"  # Name of the h-index file
    h_index_path = os.path.join(h_index_folder, h_index_file)

    if os.path.exists(h_index_path):
        h_index_df = pd.read_excel(h_index_path)
        h_index_dict = dict(zip(h_index_df['Journal'], h_index_df['Normalized H-Index']))
        journal_list = h_index_df['Journal'].tolist()  # List of known journal names
    else:
        st.error(f"Normalized H-Index file not found at {h_index_path}. Please ensure the file exists.")
        return

    # Load the keyword relevance file from the resources folder
    keyword_relevance_file = "keyword_relevance.xlsx"  # Name of the keyword relevance file
    keyword_relevance_path = os.path.join(h_index_folder, keyword_relevance_file)

    if os.path.exists(keyword_relevance_path):
        keyword_relevance_df = pd.read_excel(keyword_relevance_path)
        keyword_weights = dict(zip(keyword_relevance_df['Keyword'], keyword_relevance_df['Relevance Weight']))
    else:
        st.error(f"Keyword Relevance file not found at {keyword_relevance_path}. Please ensure the file exists.")
        return

    # Upload user document
    st.header("Upload Your Document")
    user_file = st.file_uploader("Upload your document (Excel)", type=["xlsx"])

    if user_file:
        # Read the uploaded file into a DataFrame
        df = pd.read_excel(user_file)

        # Ensure the user file has the required columns
        required_columns = ['Title', 'Abstract', 'Publication Year', 'Journal']
        if all(col in df.columns for col in required_columns):
            # Convert all relevant columns to strings and handle NaN values
            df['Title'] = df['Title'].astype(str)
            df['Abstract'] = df['Abstract'].astype(str)
            df['Journal'] = df['Journal'].astype(str)
            df['Publication Year'] = pd.to_numeric(df['Publication Year'], errors='coerce')  # Ensure numeric

            # Perform fuzzy matching for journal names
            df['Matched Journal'] = df['Journal'].apply(lambda x: fuzzy_match_journal(x, journal_list))

            # Look up normalized h-index values for each journal (scaled 0-10)
            df['NHI'] = df['Matched Journal'].map(h_index_dict)

            # Check for missing h-index values
            if df['NHI'].isnull().any():
                st.warning("Warning: Some journals do not have a normalized h-index value in the source document.")

            # Calculate relevance for each row
            df['Relevance'] = df['Title'].apply(lambda x: calculate_relevance(x, keyword_weights))

            # Calculate recency score for each row
            df['Recency'] = df['Publication Year'].apply(calculate_recency)

            # Extract p-value scores from the Abstract column
            df['p-value Score'] = df['Abstract'].apply(extract_p_value_score)

            # Calculate study design score for each row
            df['Study Design'] = df.apply(lambda row: calculate_study_design(row['Title'], row['Abstract']), axis=1)

            # Calculate the Total Score for each row using the updated formula
            df['Total Score'] = (
                weights['w1'] * df['Relevance'] +
                weights['w2'] * df['NHI'] +
                weights['w3'] * df['Recency'] +
                weights['w4'] * df['Study Design'] +
                weights['w5'] * df['p-value Score']
            )

            # Display results
            st.header("Evaluation Results")
            st.write(df)

            # Download results
            st.header("Download Results")
            output_file = 'scored_data.xlsx'
            df.to_excel(output_file, index=False)
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Scored Data",
                    data=file,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("The uploaded document is missing required columns. Please ensure it contains the following columns: Title, Abstract, Publication Year, Journal.")

# Run the Streamlit app
if __name__ == "__main__":
    main()