import streamlit as st
import pandas as pd
import joblib
import ast
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import csv
import io
import fitz  # PyMuPDF for PDFs
from docx import Document  # python-docx for Word

# Load Google credentials from Streamlit Cloud Secrets
import os
from google.oauth2 import service_account
import json
from sklearn.exceptions import NotFittedError


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

credentials_json = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
credentials = service_account.Credentials.from_service_account_info(
    credentials_json,
    scopes=SCOPES
)


# Google Sheets setup
client = gspread.authorize(credentials)
sheet = client.open_by_key("1VS9EOMcn6SYjjOuhEr5axqfyJdg5ho1Un3btojDcoWc").sheet1

# Load the saved Naive Bayes model and TF-IDF vectorizer for predictions
model = joblib.load("naïve_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


    
# Load the book dataset
books_df = pd.read_csv("961.csv")
books_df['description'] = books_df['description'].fillna("")
books_df.columns = books_df.columns.str.strip()
books_df['genres'] = books_df['genres'].fillna("")
books_df['genres'] = books_df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
books_df['genre'] = books_df['genres'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
book_vectors = vectorizer.transform(books_df['description'])

# Session variables
if "book_titles" not in st.session_state:
    st.session_state.book_titles = []
if "preferences" not in st.session_state:
    st.session_state.preferences = []
if "predicted_genre" not in st.session_state:
    st.session_state.predicted_genre = ""
if "description" not in st.session_state:
    st.session_state.description = ""
if "user_history" not in st.session_state:
    st.session_state.user_history = []

# UI
st.title("📚 Book Genre Classification Using Text Analysis In Python")
user_id = st.text_input("👤 Enter your User ID", value="U1")

# File uploader for Word and PDF
uploaded_file = st.file_uploader("📂 Upload a Word or PDF file", type=["pdf", "docx"])

def extract_text_from_pdf(uploaded_file):
    # Open the PDF from the uploaded file's binary data directly
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Using `stream` to handle in-memory data
    
    text = ""
    for page in doc:
        text += page.get_text()
    
    return text

def extract_text_from_word(uploaded_file):
    doc = Document(uploaded_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"  # Fixed unterminated string error
    return text

# Text area to show description or upload a file
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.session_state.description = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        st.session_state.description = extract_text_from_word(uploaded_file)

description = st.text_area("📝 Describe a Description you like:", value=st.session_state.description, height=200)

# Predict button
if st.button("🔍 Predict"):
    if not description.strip():
        st.warning("Please enter a book description.")
    else:
        st.session_state.description = description

        # Transform the description input using the vectorizer
        x_input = vectorizer.transform([description])  # This line is sufficient

        # Now pass the transformed input to the model for prediction
        st.session_state.predicted_genre = model.predict(x_input)[0]  # Model prediction

        confidence = model.predict_proba(x_input).max() * 100
        st.success(f"🎯 Predicted Genre: `{st.session_state.predicted_genre}` ({confidence:.1f}% confidence)")

        books_df['similarity'] = cosine_similarity(x_input, book_vectors).flatten()
        filtered = books_df[books_df['genre'] == st.session_state.predicted_genre]
        recommended = filtered.sort_values(by='similarity', ascending=False).head(5)

        st.session_state.book_titles = []
        st.session_state.preferences = []

        if not recommended.empty:
            st.subheader("📖 Feedback for Recommended Books")
            for idx, row in recommended.iterrows():
                st.session_state.book_titles.append(row["title"])
                st.session_state.preferences.append("")


# Show radio options only if books exist
if st.session_state.book_titles:
    st.markdown("### 📋 Select 'Liked' or 'Disliked' for each book:")

    for idx, book in enumerate(st.session_state.book_titles):
        col1, col2 = st.columns([4, 2])
        with col1:
            st.markdown(f"**{idx+1}. {book}**")
        with col2:
            st.session_state.preferences[idx] = st.radio(
    f"Book {idx+1} Preference", ["Liked", "Disliked"], key=f"radio_{idx}", horizontal=True, label_visibility="collapsed"
)


    if st.button("✅ Submit Preference"):
        if any(p == "" for p in st.session_state.preferences):
            st.warning("⚠️ Please select 'Liked' or 'Disliked' for all books before submitting.")
            st.stop()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_rows = []
        for i in range(len(st.session_state.book_titles)):
            row = [
                user_id if i == 0 else "",
                st.session_state.predicted_genre if i == 0 else "",
                st.session_state.book_titles[i],
                st.session_state.preferences[i],
                timestamp if i == 0 else ""
            ]
            all_rows.append(row)

        with st.spinner("📤 Submitting your preferences to Google Sheet..."):
            try:
                sheet.append_rows(all_rows)
                st.success("✅ Preferences saved in vertical format!")

                # Store user history for later reference
                st.session_state.user_history.append({
                    "predicted_genre": st.session_state.predicted_genre,
                    "book_titles": st.session_state.book_titles,
                    "preferences": st.session_state.preferences,
                    "timestamp": timestamp
                })
            except Exception as e:
                st.error("❌ Could not write to Google Sheet!")
                st.error(f"Error: {e}")

# Download recommendation as CSV
def create_csv():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["User ID", "Predicted Genre", "Book Title", "Preference", "Timestamp"])
    for book, pref in zip(st.session_state.book_titles, st.session_state.preferences):
        writer.writerow([user_id, st.session_state.predicted_genre, book, pref, timestamp])
    return csv_buffer.getvalue()

st.download_button(
    label="📥 Download Recommendations",
    data=create_csv(),
    file_name="book_recommendations.csv",
    mime="text/csv"
)


# Reset button to clear state
if st.button("🔁 Reset"):
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Specifically clear the description and uploaded file state
    st.session_state.description = ""
    st.session_state.book_titles = []
    st.session_state.preferences = []
    st.session_state.predicted_genre = ""
    
    # Use st.empty to clear the file uploader state
    st.file_uploader("📂 Upload a Word or PDF file", type=["pdf", "docx"], key="clear_file_uploader")

    #Rerun the app to reset UI and state
    st.rerun()
