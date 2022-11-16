import nltk
import pickle
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


import re
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet=WordNetLemmatizer()

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/spamclassifier.pkl","rb") as f:
    model = pickle.load(f)

def predict_label(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [wordnet.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)

    prediction=  model.predict([text])

    return ('ham' if prediction[0]==0 else 'spam')