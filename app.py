import streamlit as st
import pickle as pkl
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

# Load the model
clf = pkl.load(open('model.pkl', 'rb'))
tfidf = pkl.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    # Remove URLs
    cleanTxt = re.sub(r'http\S+', '', txt)
    # Remove email addresses or words like "@gmail.com"
    cleanTxt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b', '', cleanTxt)
    # Remove special characters
    cleanTxt = re.sub(r'[^a-zA-Z0-9\s]', '', cleanTxt)
    # Remove RT, cc
    cleanTxt = re.sub(r'\bRT\b|\bcc\b', '', cleanTxt)
    # Remove hashtags
    cleanTxt = re.sub(r'#\S+', '', cleanTxt)
    # Remove mentions
    cleanTxt = re.sub(r'@\S+', '', cleanTxt)
    # Remove non-ASCII characters
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    # Collapse extra spaces
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()
    return cleanTxt

categorical_mapping = {
    0: 'Advocate',
    1: 'Arts',
    2: 'Automation Testing',
    4: 'Business Analyst',
    5: 'Civil Engineer',
    8: 'DevOps Engineer',
    11: 'Electrical Engineering',
    14: 'Health and fitness',
    15: 'Java Developer',
    16: 'Mechanical Engineer',
    17: 'Network Security Engineer',
    18: 'Operations Manager',
    20: 'Python Developer',
    21: 'SAP Developer',
    22: 'Sales',
    24: 'Web Designing',
    12: 'HR'
}

# Web app
def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader('Upload your resume', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        transformed_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(transformed_resume)[0]

        category_name = categorical_mapping.get(prediction_id, 'Unknown')
        st.write('Predicted Category:', category_name)

if __name__ == '__main__':
    main()