"""
Text Preprocessing Module
Implements comprehensive text cleaning and preprocessing pipeline
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
import contractions

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for consumer complaints
    """
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_special_chars: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 stem: bool = False,
                 min_word_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize text preprocessor
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters
            remove_numbers: Remove numeric values
            remove_stopwords: Remove stop words
            lemmatize: Apply lemmatization
            stem: Apply stemming (not used if lemmatize=True)
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem and not lemmatize else None
        
        logger.info("TextPreprocessor initialized")
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions (e.g., don't -> do not)
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        try:
            return contractions.fix(text)
        except:
            # Fallback if contractions library fails
            return text
    
    def clean_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', '', text)
        return text
    
    def clean_emails(self, text: str) -> str:
        """
        Remove email addresses from text
        
        Args:
            text: Input text
            
        Returns:
            Text without email addresses
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def clean_special_chars(self, text: str) -> str:
        """
        Remove special characters and punctuation
        
        Args:
            text: Input text
            
        Returns:
            Text without special characters
        """
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text
    
    def clean_numbers(self, text: str) -> str:
        """
        Remove numbers from text
        
        Args:
            text: Input text
            
        Returns:
            Text without numbers
        """
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered token list
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens
        
        Args:
            tokens: List of tokens
            
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def filter_short_words(self, tokens: List[str]) -> List[str]:
        """
        Remove words shorter than min_word_length
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        return [token for token in tokens if len(token) >= self.min_word_length]
    
    def preprocess(self, text: str, return_tokens: bool = False) -> str:
        """
        Apply full preprocessing pipeline to text
        
        Args:
            text: Input text
            return_tokens: If True, return list of tokens instead of string
            
        Returns:
            Preprocessed text or list of tokens
        """
        if not isinstance(text, str):
            return "" if not return_tokens else []
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = self.clean_urls(text)
        
        # Remove emails
        if self.remove_emails:
            text = self.clean_emails(text)
        
        # Remove special characters
        if self.remove_special_chars:
            text = self.clean_special_chars(text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self.clean_numbers(text)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Filter short words
        tokens = self.filter_short_words(tokens)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.filter_stopwords(tokens)
        
        # Lemmatization
        if self.lemmatize and self.lemmatizer:
            tokens = self.apply_lemmatization(tokens)
        
        # Stemming (only if not lemmatizing)
        if self.stem and self.stemmer and not self.lemmatize:
            tokens = self.apply_stemming(tokens)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_dataframe(self, 
                            df: pd.DataFrame, 
                            text_column: str = 'text',
                            new_column: str = 'processed_text') -> pd.DataFrame:
        """
        Apply preprocessing to a DataFrame column
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            new_column: Name for preprocessed text column
            
        Returns:
            DataFrame with preprocessed text column
        """
        logger.info(f"Preprocessing {len(df)} texts...")
        
        df = df.copy()
        df[new_column] = df[text_column].apply(self.preprocess)
        
        # Remove empty processed texts
        original_len = len(df)
        df = df[df[new_column].str.strip() != '']
        removed = original_len - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with empty processed text")
        
        logger.info(f"Preprocessing complete. Final dataset size: {len(df)}")
        
        return df


def get_text_statistics(df: pd.DataFrame, 
                        text_column: str = 'text') -> pd.DataFrame:
    """
    Calculate text statistics for analysis
    
    Args:
        df: DataFrame containing text
        text_column: Name of text column
        
    Returns:
        DataFrame with text statistics
    """
    df = df.copy()
    
    df['char_count'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    df['avg_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(word) for word in str(x).split()])
    )
    df['stopword_count'] = df[text_column].apply(
        lambda x: len([w for w in str(x).lower().split() 
                      if w in stopwords.words('english')])
    )
    
    return df


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_stopwords=True,
        lemmatize=True
    )
    
    sample_text = "I've been trying to contact XXXX about errors on my credit report at http://example.com. Email: test@example.com"
    
    processed = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
