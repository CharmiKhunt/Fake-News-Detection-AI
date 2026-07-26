import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only the first time)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


def clean_text(text):
    if not isinstance(text, str):
        text = str(text or "")

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split into words
    words = text.split()

    # Remove stopwords and stem each word
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    # Join back into one string
    return " ".join(words)