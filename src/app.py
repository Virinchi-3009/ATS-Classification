from flask import Flask, render_template, request, jsonify
import re
import joblib
import fitz  # PyMuPDF

app = Flask(__name__)

# Load your pre-trained ML model and vectorizer
pipeline = joblib.load('C:/Users/aarad/OneDrive/Desktop/ATS- Machine Learning/models/classifier.pkl')

# Define a sample job description
JOB_DESCRIPTION = "We are looking for a skilled software engineer with experience in Python, Java, and web development."

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text.strip()

# Function to analyze resume against job description
def analyze_resume_locally(job_description, resume_text):
    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume_text)

    job_keywords = set(job_description.split())
    resume_keywords = set(resume_text.split())

    matched_keywords = job_keywords.intersection(resume_keywords)
    missing_keywords = job_keywords.difference(resume_keywords)

    match_percentage = len(matched_keywords) / len(job_keywords) * 100 if len(job_keywords) > 0 else 0

    return {
        "matched_keywords": list(matched_keywords),
        "missing_keywords": list(missing_keywords),
        "match_percentage": match_percentage
    }

# Function to predict resume quality based on match percentage
def predict_resume_quality(match_percentage):
    if match_percentage > 40:
        return "Good"
    else:
        return "Needs Improvement"

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    job_description = request.form.get('job_description', JOB_DESCRIPTION)

    resume_text = extract_text_from_pdf(file)
    if not resume_text:
        return jsonify({'error': 'Failed to extract text from PDF'}), 500

    # Analyze resume and predict quality
    local_analysis = analyze_resume_locally(job_description, resume_text)
    prediction = predict_resume_quality(local_analysis['match_percentage'])

    return jsonify({
        'match_percentage': local_analysis['match_percentage'],
        'predicted_quality': prediction
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
