import re
import fitz  # PyMuPDF
import pandas as pd

def preprocess_text(text):
    """Preprocess the text by converting to lowercase and removing non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def extract_text_from_pdf(file):
    """Extract text from a PDF file using PyMuPDF."""
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text.strip()

def extract_features(text):
    """Extract features from the text, such as word count."""
    features = {
        'word_count': len(text.split()),
        # Add more features as needed
    }
    return features

def load_labels(csv_path):
    """Load labels from a CSV file."""
    return pd.read_csv(csv_path)

# Sample job description for testing purposes
JOB_DESCRIPTION = "We are looking for a skilled software engineer with experience in Python, Java, and web development."
