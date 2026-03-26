from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import nltk
from nltk.corpus import stopwords
import re

# Load stopwords safely
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# Extract text from PDF
def extract_text(file):
    pdf = PyPDF2.PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text.lower()


# Calculate similarity score
def calculate_score(resume, job_desc):
    texts = [resume, job_desc]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Clean words using regex + remove stopwords
    resume_words = set([
        word for word in re.findall(r'\b\w+\b', resume)
        if word not in stop_words
    ])

    job_words = set([
        word for word in re.findall(r'\b\w+\b', job_desc.lower())
        if word not in stop_words
    ])

    matched = list(resume_words.intersection(job_words))

    return round(similarity * 100, 2), matched[:10]