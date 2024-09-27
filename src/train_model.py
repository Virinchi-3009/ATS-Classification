import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
data = pd.read_csv('C:/Users/aarad/OneDrive/Desktop/ATS- Machine Learning/data/labels.csv')

# Prepare data
X = data['Resume File']  # Column with resume text
y = data['Label']  # Column with labels

# Preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

X = X.apply(preprocess_text)  # Apply preprocessing to each resume text

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # Text vectorization
    ('classifier', MultinomialNB())     # Naive Bayes classifier
])

pipeline.fit(X_train, y_train)

# Save the entire pipeline (includes both vectorizer and classifier)
model_path = 'C:/Users/aarad/OneDrive/Desktop/ATS- Machine Learning/models/classifier.pkl'
joblib.dump(pipeline, model_path)

print(f"Model saved to {model_path}")
