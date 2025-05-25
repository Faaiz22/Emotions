import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor with required NLTK components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords_and_lemmatize(self, text):
        """
        Remove stopwords and lemmatize text.
        
        Args:
            text (str): Cleaned text input
            
        Returns:
            str: Text with stopwords removed and words lemmatized
        """
        if not text:
            return ""
        
        words = text.split()
        
        # Remove stopwords and lemmatize
        processed_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(processed_words)
    
    def preprocess_text_column(self, texts):
        """
        Apply full text preprocessing pipeline to a series of texts.
        
        Args:
            texts (pd.Series): Series of text data
            
        Returns:
            pd.Series: Preprocessed text data
        """
        print("Starting text preprocessing...")
        
        # Clean text
        print("- Cleaning text...")
        cleaned_texts = texts.apply(self.clean_text)
        
        # Remove stopwords and lemmatize
        print("- Removing stopwords and lemmatizing...")
        processed_texts = cleaned_texts.apply(self.remove_stopwords_and_lemmatize)
        
        print("Text preprocessing completed!")
        return processed_texts
    
    def create_tfidf_features(self, texts, max_features=10000, ngram_range=(1, 2)):
        """
        Create TF-IDF features from preprocessed text.
        
        Args:
            texts (pd.Series): Preprocessed text data
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to consider
            
        Returns:
            scipy.sparse.matrix: TF-IDF feature matrix
        """
        print("Creating TF-IDF features...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        
        # Fit and transform the text data
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF features created! Shape: {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def save_vectorizer(self, filepath):
        """
        Save the trained TF-IDF vectorizer.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer has not been trained yet!")
        
        joblib.dump(self.tfidf_vectorizer, filepath)
        print(f"TF-IDF vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath):
        """
        Load a pre-trained TF-IDF vectorizer.
        
        Args:
            filepath (str): Path to the saved vectorizer
        """
        self.tfidf_vectorizer = joblib.load(filepath)
        print(f"TF-IDF vectorizer loaded from {filepath}")
    
    def load_and_preprocess_data(self, csv_path, text_column, label_column, test_size=0.2, random_state=42):
        """
        Load data from CSV and perform complete preprocessing pipeline.
        
        Args:
            csv_path (str): Path to the CSV file
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            test_size (float): Proportion of data to use for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_matrix, labels)
        """
        print(f"Loading data from {csv_path}...")
        
        # Load the dataset
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        print(f"Missing values in {text_column}: {df[text_column].isna().sum()}")
        print(f"Missing values in {label_column}: {df[label_column].isna().sum()}")
        
        # Remove rows with missing values
        df = df.dropna(subset=[text_column, label_column])
        print(f"Dataset after removing missing values: {df.shape}")
        
        # Display label distribution
        print("\nLabel distribution:")
        print(df[label_column].value_counts().sort_index())
        
        # Preprocess text
        processed_texts = self.preprocess_text_column(df[text_column])
        
        # Create TF-IDF features
        tfidf_matrix = self.create_tfidf_features(processed_texts)
        
        # Prepare labels
        labels = df[label_column].values
        
        # Split the data
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, tfidf_matrix, labels