import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textstat
import re

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def analyze_tabular_data(df):
    """
    Generate comprehensive analysis of tabular data
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
        
    Returns:
    --------
    analysis : dict
        Dictionary containing various analyses
    """
    analysis = {}
    
    # Basic info
    analysis['shape'] = df.shape
    analysis['columns'] = list(df.columns)
    analysis['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    analysis['missing_values'] = {col: {'count': count, 'percentage': pct} 
                                 for col, count, pct in zip(missing_values.index, 
                                                          missing_values.values, 
                                                          missing_percentage.values)}
    
    # Numerical stats
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        analysis['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Correlation matrix
        if len(numerical_cols) > 1:
            analysis['correlation'] = df[numerical_cols].corr().to_dict()
    
    # Categorical stats
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        analysis['categorical_stats'] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            unique_count = df[col].nunique()
            analysis['categorical_stats'][col] = {
                'unique_count': unique_count,
                'top_values': value_counts
            }
    
    # Detect potential outliers in numerical columns
    analysis['potential_outliers'] = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outlier_count > 0:
            analysis['potential_outliers'][col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100
            }
    
    # Data dimensions analysis (PCA)
    if len(numerical_cols) > 2:
        try:
            pca = PCA()
            pca_data = df[numerical_cols].dropna()
            if len(pca_data) > 0:
                pca.fit(pca_data)
                analysis['pca'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist()
                }
        except:
            pass
    
    return analysis

def analyze_text_data(text):
    """
    Generate comprehensive analysis of text data
    
    Parameters:
    -----------
    text : str
        The text to analyze
        
    Returns:
    --------
    analysis : dict
        Dictionary containing various text analyses
    """
    analysis = {}
    
    # Basic stats
    analysis['char_count'] = len(text)
    analysis['word_count'] = len(text.split())
    analysis['sentence_count'] = len(re.split(r'[.!?]+', text))
    analysis['paragraph_count'] = len(text.split('\n\n'))
    
    # Readability metrics
    analysis['readability'] = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'smog_index': textstat.smog_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text)
    }
    
    # Word frequency analysis
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    word_freq = {}
    for word in filtered_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Sort by frequency
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    analysis['top_words'] = dict(list(sorted_word_freq.items())[:50])
    
    return analysis

def generate_data_summary(data, data_type):
    """
    Generate a human-readable summary of the data
    
    Parameters:
    -----------
    data : pd.DataFrame or str
        The data to summarize
    data_type : str
        Type of data ('tabular' or 'text')
        
    Returns:
    --------
    summary : str
        Human-readable summary
    """
    if data_type == 'tabular':
        rows, cols = data.shape
        num_missing = data.isna().sum().sum()
        missing_pct = (num_missing / (rows * cols)) * 100
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        summary = f"""
## Data Summary

This dataset contains **{rows:,} rows** and **{cols} columns**.

- **Missing values**: {num_missing:,} ({missing_pct:.2f}% of all values)
- **Numerical columns**: {len(numerical_cols)} ({', '.join(numerical_cols[:5])}{"..." if len(numerical_cols) > 5 else ""})
- **Categorical columns**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{"..." if len(categorical_cols) > 5 else ""})

### Key Insights:
"""
        
        # Add insights about numerical columns
        if len(numerical_cols) > 0:
            for col in numerical_cols[:3]:  # Show insights for up to 3 numerical columns
                mean_val = data[col].mean()
                median_val = data[col].median()
                min_val = data[col].min()
                max_val = data[col].max()
                summary += f"""
- **{col}**: Ranges from {min_val:,.2f} to {max_val:,.2f}, with mean {mean_val:,.2f} and median {median_val:,.2f}
"""
        
        # Add insights about categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:3]:  # Show insights for up to 3 categorical columns
                unique_count = data[col].nunique()
                top_value = data[col].value_counts().idxmax()
                top_pct = (data[col].value_counts().max() / len(data)) * 100
                summary += f"""
- **{col}**: Has {unique_count} unique values, with '{top_value}' being the most common ({top_pct:.2f}% of data)
"""
        
        return summary
        
    elif data_type == 'text':
        char_count = len(data)
        word_count = len(data.split())
        sentence_count = len(re.split(r'[.!?]+', data))
        paragraph_count = len(data.split('\n\n'))
        flesch_score = textstat.flesch_reading_ease(data)
        
        # Determine readability level
        if flesch_score >= 90:
            readability = "Very Easy"
        elif flesch_score >= 80:
            readability = "Easy"
        elif flesch_score >= 70:
            readability = "Fairly Easy"
        elif flesch_score >= 60:
            readability = "Standard"
        elif flesch_score >= 50:
            readability = "Fairly Difficult"
        elif flesch_score >= 30:
            readability = "Difficult"
        else:
            readability = "Very Difficult"
        
        summary = f"""
## Text Summary

This document contains:
- **{char_count:,} characters**
- **{word_count:,} words**
- **{sentence_count:,} sentences**
- **{paragraph_count:,} paragraphs**

The text has a Flesch Reading Ease score of **{flesch_score:.2f}**, which is classified as "**{readability}**" to read.

### Content Preview:
{data[:300]}...
"""
        
        # Add additional insights about vocabulary
        words = word_tokenize(data.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        unique_words = len(set(filtered_words))
        lexical_diversity = unique_words / max(1, len(filtered_words))
        
        summary += f"""
### Vocabulary Analysis:
- **Unique meaningful words**: {unique_words:,}
- **Lexical diversity**: {lexical_diversity:.4f} (higher means more diverse vocabulary)
"""
        
        # Add common words section
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
                
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
        top_5_words = list(sorted_word_freq.keys())[:5]
        
        if top_5_words:
            summary += f"""
### Most Common Words:
{', '.join([f"**{word}**" for word in top_5_words])}
"""
        
        return summary