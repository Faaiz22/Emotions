import os
import sys
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import joblib
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Emotion mapping for the emotions dataset
EMOTION_MAPPING = {
    0: 'Neutral/Factual Statement',
    1: 'Mild Positive/Contentment',
    2: 'Slight Positivity/Nostalgia/Curiosity',
    3: 'Mild Negativity/Uncertainty/Stress',
    4: 'Strong Negative Emotion/Distress/Sadness',
    5: 'Extreme Distress/Overwhelming Emotion'
}

class FastEmotionClassifier:
    def __init__(self):
        """Initialize the optimized emotion classifier."""
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.feature_selector = None
        
    def clean_and_preprocess_text(self, text):
        """Fast text preprocessing."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        # Quick tokenization and stopword removal
        words = text.split()
        processed_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(processed_words)
    
    def load_and_prepare_data(self, csv_path, text_column='text', label_column='label', 
                             sample_size=None, max_features=5000):
        """
        Load and prepare data with optimizations for speed.
        
        Args:
            csv_path (str): Path to CSV file
            text_column (str): Name of text column
            label_column (str): Name of label column
            sample_size (int): Limit dataset size for faster training
            max_features (int): Reduce TF-IDF features for speed
        """
        print(f"Loading data from {csv_path}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Sample data if specified for faster training
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled dataset shape: {df.shape}")
        
        # Remove missing values
        df = df.dropna(subset=[text_column, label_column])
        print(f"Dataset after cleaning: {df.shape}")
        
        # Display label distribution
        print("\nLabel distribution:")
        print(df[label_column].value_counts().sort_index())
        
        # Preprocess text data
        print("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self.clean_and_preprocess_text)
        
        # Create TF-IDF features with reduced dimensionality
        print(f"Creating TF-IDF features (max_features={max_features})...")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df[label_column].values
        
        # Optional: Further feature selection for speed
        if max_features > 2000:
            print("Applying feature selection...")
            self.feature_selector = SelectKBest(chi2, k=min(2000, X.shape[1]))
            X = self.feature_selector.fit_transform(X, y)
            print(f"Features after selection: {X.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def quick_kernel_comparison(self, X_train, y_train, cv_folds=3):
        """
        Quick comparison of SVM kernels with limited parameters.
        """
        print("\nQuick kernel comparison...")
        
        kernels_configs = {
            'linear': {'kernel': 'linear', 'C': 1},
            'rbf': {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
            'poly': {'kernel': 'poly', 'C': 1, 'degree': 3, 'gamma': 'scale'}
        }
        
        results = {}
        
        for kernel_name, params in kernels_configs.items():
            print(f"Testing {kernel_name} kernel...")
            start_time = time.time()
            
            svm = SVC(**params, random_state=42)
            cv_scores = cross_val_score(svm, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            end_time = time.time()
            
            results[kernel_name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'time': end_time - start_time
            }
            
            print(f"{kernel_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}) - {end_time - start_time:.2f}s")
        
        return results
    
    def fast_grid_search(self, X_train, y_train, cv_folds=3):
        """
        Optimized grid search with reduced parameter space.
        """
        print("\nFast grid search optimization...")
        
        # Reduced parameter grid for speed
        param_grid = [
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10]
            },
            {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1]
            }
        ]
        
        svm = SVC(random_state=42)
        
        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best CV score: {self.best_score:.4f}")
        print(f"Grid search time: {end_time - start_time:.2f} seconds")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'time': end_time - start_time
        }
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model."""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        print("\nEvaluating model...")
        y_pred = self.best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        target_names = [EMOTION_MAPPING.get(i, f'Class {i}') for i in sorted(set(y_test))]
        class_report = classification_report(y_test, y_pred, target_names=target_names)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nDetailed Classification Report:\n{class_report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }
    
    def predict_emotion(self, text):
        """Predict emotion for given text."""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess text
        processed_text = self.clean_and_preprocess_text(text)
        
        # Transform to TF-IDF
        tfidf_features = self.vectorizer.transform([processed_text])
        
        # Apply feature selection if used
        if self.feature_selector:
            tfidf_features = self.feature_selector.transform(tfidf_features)
        
        # Predict
        prediction = self.best_model.predict(tfidf_features)[0]
        emotion = EMOTION_MAPPING.get(prediction, f'Unknown ({prediction})')
        
        return emotion, prediction
    
    def save_model_components(self, model_prefix="fast_emotion_model"):
        """Save all model components."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f"{model_prefix}_{timestamp}.pkl"
        joblib.dump(self.best_model, model_path)
        
        # Save vectorizer
        vectorizer_path = f"vectorizer_{timestamp}.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save feature selector if used
        if self.feature_selector:
            selector_path = f"feature_selector_{timestamp}.pkl"
            joblib.dump(self.feature_selector, selector_path)
            print(f"Feature selector saved: {selector_path}")
        
        print(f"Model saved: {model_path}")
        print(f"Vectorizer saved: {vectorizer_path}")
        
        return model_path, vectorizer_path

def run_fast_emotion_classification(csv_path, sample_size=10000, max_features=3000):
    """
    Run the optimized emotion classification pipeline.
    
    Args:
        csv_path (str): Path to the emotions CSV file
        sample_size (int): Limit dataset size for faster training
        max_features (int): Limit TF-IDF features for speed
    """
    print("="*60)
    print("FAST EMOTION CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"Optimizations applied:")
    print(f"- Sample size limit: {sample_size}")
    print(f"- Max TF-IDF features: {max_features}")
    print(f"- Reduced parameter grid")
    print(f"- 3-fold CV instead of 5-fold")
    print("="*60)
    
    classifier = FastEmotionClassifier()
    
    try:
        # Step 1: Load and prepare data
        print("\nSTEP 1: DATA PREPARATION")
        print("-" * 30)
        X_train, X_test, y_train, y_test = classifier.load_and_prepare_data(
            csv_path, 
            sample_size=sample_size, 
            max_features=max_features
        )
        
        # Step 2: Quick kernel comparison
        print("\nSTEP 2: KERNEL COMPARISON")
        print("-" * 30)
        kernel_results = classifier.quick_kernel_comparison(X_train, y_train)
        
        # Step 3: Grid search optimization
        print("\nSTEP 3: MODEL OPTIMIZATION")
        print("-" * 30)
        training_results = classifier.fast_grid_search(X_train, y_train)
        
        # Step 4: Model evaluation
        print("\nSTEP 4: MODEL EVALUATION")
        print("-" * 30)
        evaluation_results = classifier.evaluate_model(X_test, y_test)
        
        # Step 5: Save models
        print("\nSTEP 5: SAVING MODELS")
        print("-" * 30)
        model_path, vectorizer_path = classifier.save_model_components()
        
        # Step 6: Demo predictions
        print("\nSTEP 6: DEMO PREDICTIONS")
        print("-" * 30)
        sample_texts = [
            "The weather is nice today",
            "I'm feeling great about this project",
            "This reminds me of my childhood",
            "I'm worried about the deadline",
            "I'm completely devastated by this news",
            "This is absolutely overwhelming"
        ]
        
        for text in sample_texts:
            emotion, label = classifier.predict_emotion(text)
            print(f"'{text}' → {emotion}")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return classifier, {
            'kernel_results': kernel_results,
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise e

# Example usage
if __name__ == "__main__":
    # Configuration for fast execution
    CSV_PATH = "emotions.csv"  # Update path as needed
    SAMPLE_SIZE = 5000  # Use smaller sample for very fast results
    MAX_FEATURES = 2000  # Reduce features for speed
    
    # Check if file exists
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found!")
        print("Please upload your emotions.csv file first.")
    else:
        # Run the fast pipeline
        classifier, results = run_fast_emotion_classification(
            CSV_PATH, 
            sample_size=SAMPLE_SIZE, 
            max_features=MAX_FEATURES
        )