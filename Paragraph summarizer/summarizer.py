import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Tokenize text into words
    words = word_tokenize(text)
    # Remove stop words and punctuation
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    # Stem words
    words = [stemmer.stem(word) for word in words]
    return words

def calculate_sentence_scores(sentences, tfidf_model, dictionary):
    scores = []
    for sentence in sentences:
        # Preprocess sentence
        words = preprocess_text(sentence)
        # Convert sentence to TF-IDF vector
        bow = dictionary.doc2bow(words)
        tfidf_vector = tfidf_model[bow]
        # Calculate sentence score as sum of TF-IDF scores for all words
        score = sum(score for _, score in tfidf_vector)
        scores.append(score)
    return scores

def summarize_text(text, num_sentences=3):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Preprocess sentences
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Create dictionary and TF-IDF model
    dictionary = Dictionary(preprocessed_sentences)
    corpus = [dictionary.doc2bow(words) for words in preprocessed_sentences]
    tfidf_model = TfidfModel(corpus)

    # Calculate sentence scores
    sentence_scores = calculate_sentence_scores(sentences, tfidf_model, dictionary)

    # Select top-ranked sentences based on scores
    ranked_sentences = sorted(((score, sentence) for score, sentence in zip(sentence_scores, sentences)), reverse=True)
    summary_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]

    # Join summary sentences into a single summary
    summary = ' '.join(summary_sentences)
    return summary

# Read text from a file
# brief taken from "https://en.wikipedia.org/wiki/Artificial_intelligence"

file_path =("C:\\Users\\ILENE\\Downloads\\word predictor\\AI.txt")
with open(file_path, 'r') as file:
    input_text = file.read()

# Summarize input text
summary = summarize_text(input_text)
print("\nSummary:\n", summary)