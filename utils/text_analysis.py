import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import string
import re
import textstat
from collections import Counter

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess text for analysis
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
        
    Returns:
    --------
    cleaned_text : str
        Preprocessed text
    tokens : list
        List of tokens
    sentences : list
        List of sentences
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstruct cleaned text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text, tokens, sentences

def extract_key_phrases(tokens, n=2):
    """
    Extract key phrases (n-grams) from tokens
    
    Parameters:
    -----------
    tokens : list
        List of tokens
    n : int
        Size of n-grams
        
    Returns:
    --------
    ngrams : list
        List of most common n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    
    freq_dist = FreqDist(ngrams)
    return freq_dist.most_common(20)  # Return top 20 n-grams

def calculate_readability_metrics(text):
    """
    Calculate various readability metrics
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    metrics : dict
        Dictionary of readability metrics
    """
    metrics = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'smog_index': textstat.smog_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'difficult_words': textstat.difficult_words(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'gunning_fog': textstat.gunning_fog(text),
        'text_standard': textstat.text_standard(text)
    }
    
    return metrics

def summarize_text(text, ratio=0.3):
    """
    Generate a summary of the text
    
    Parameters:
    -----------
    text : str
        Text to summarize
    ratio : float
        Proportion of sentences to include in summary
        
    Returns:
    --------
    summary : str
        Summarized text
    """
    # If text is very short, return as is
    sentences = sent_tokenize(text)
    if len(sentences) <= 3:
        return text
    
    # Preprocess
    cleaned_text, tokens, _ = preprocess_text(text)
    
    # Create frequency distribution of words
    word_freq = FreqDist(tokens)
    
    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if i in sentence_scores:
                    sentence_scores[i] += word_freq[word]
                else:
                    sentence_scores[i] = word_freq[word]
    
    # Calculate number of sentences for summary
    num_sentences = max(1, int(len(sentences) * ratio))
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Sort by original order
    
    # Construct summary
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    
    return summary

def extract_sentiment(text):
    """
    Simple sentiment analysis
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    sentiment : dict
        Dictionary with positive/negative words and scores
    """
    # Load positive and negative words
    try:
        with open('utils/positive_words.txt', 'r') as f:
            positive_words = set(f.read().splitlines())
    except:
        positive_words = set(['good', 'great', 'excellent', 'positive', 'nice', 'wonderful', 'amazing', 'love', 'best', 'happy'])
        
    try:
        with open('utils/negative_words.txt', 'r') as f:
            negative_words = set(f.read().splitlines())
    except:
        negative_words = set(['bad', 'worst', 'terrible', 'negative', 'awful', 'wrong', 'horrible', 'hate', 'poor', 'sad'])
    
    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    # Count positive and negative words
    positive_count = sum(1 for word in tokens if word in positive_words)
    negative_count = sum(1 for word in tokens if word in negative_words)
    total_count = len(tokens)
    
    # Calculate scores
    if total_count > 0:
        positive_score = positive_count / total_count
        negative_score = negative_count / total_count
        compound_score = (positive_count - negative_count) / total_count
    else:
        positive_score = negative_score = compound_score = 0
    
    # Find matching words
    positive_matches = [word for word in tokens if word in positive_words]
    negative_matches = [word for word in tokens if word in negative_words]
    
    sentiment = {
        'positive_score': positive_score,
        'negative_score': negative_score,
        'compound_score': compound_score,
        'positive_words': dict(Counter(positive_matches).most_common(10)),
        'negative_words': dict(Counter(negative_matches).most_common(10))
    }
    
    return sentiment