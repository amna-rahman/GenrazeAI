
# GenrifyAI: Book Genre Classification using Text Analysis in Python

**GenrifyAI** is a web app built with **Streamlit** that predicts the genre of a book based on its description using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. By leveraging a **Naive Bayes classifier** and **TF-IDF vectorization**, the app classifies book genres and provides users with relevant recommendations.

---

## Features
- **Real-time genre prediction** based on book descriptions.
- **Upload PDF/Word files** containing book descriptions for genre classification.
- **User feedback system** to rate predictions and save preferences.
- **Integration with Google Sheets** to save user preferences for future reference.
- **Interactive history tracking** of previous predictions within the session.

---

## Installation

### Step 1: **Clone the repository**
```bash
git clone https://github.com/your-username/genrify.git
cd genrify
```

### Step 2: **Install dependencies**
```bash
pip install -r requirements.txt
```

### Step 3: **Set up Google Sheets API**
-  Go to the Google Developer Console.
-  Enable the **Google Sheets API** for your project.
-  Download the **credentials.json** file.
-  Place the **credentials.json** file in the project directory.

### Step 4: **Install additional dependencies for file handling**
```bash
pip install python-docx PyMuPDF
```

---

## Usage

### Step 1: **Launch the app**
```bash
streamlit run app.py
```

### Step 2: **Interact with the app**
- **Enter Description**: Type or paste a book description.
- **Upload File**: Upload a PDF or Word file containing the book description.
- **Prediction**: Click the **Predict** button to classify the genre.
- **Feedback**: Mark predictions as **Liked** or **Disliked**.
- **Reset**: Clear inputs and start a new session.

---

## Requirements
- Python 3.x
- Streamlit
- scikit-learn
- pandas
- gspread (for Google Sheets)
- nltk (for text preprocessing)
- PyMuPDF (for PDF extraction)
- python-docx (for Word file extraction)

---

## Model

### **Training**
-  The model uses **TF-IDF Vectorization** for text feature extraction and a **Naive Bayes classifier** for genre prediction.
-  You can upload a labeled dataset to train a custom model.

### **Pre-trained Model**
-  **naive_bayes_model.pkl** and **tfidf_vectorizer.pkl** are used for making predictions.
-  Replace these files with your trained model if needed.

---

## Folder Structure
```
genrify/
│
├── app.py                    # Main Streamlit app file
├── requirements.txt           # List of dependencies
├── naive_bayes_model.pkl      # Pre-trained Naive Bayes model
├── tfidf_vectorizer.pkl       # Pre-trained TF-IDF vectorizer
├── credentials.json           # Google Sheets API credentials
└── data/                      # Optional folder for any data files
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements
- **Streamlit**: Easy framework for building web apps.
- **scikit-learn**: Machine learning tools for Naive Bayes and TF-IDF.
- **Google Sheets API**: For saving user preferences.
- **nltk**: Natural language processing for stopword removal and tokenization.
- **PyMuPDF & python-docx**: For PDF and Word file parsing.

---

## How to Contribute
Feel free to fork, contribute, or suggest improvements. Open an issue if you encounter bugs or have feature requests.

### **Instructions to Save as a File**:
1. **Copy the content** above.
2. **Create a new file** named `README.md` in the root directory of your project.
3. **Paste the content** into the `README.md` file.
4. **Save** and **push** this file to your GitHub repository.

This will make your **project documentation** available and well-structured when published on GitHub.

Let me know if you need any further modifications!
