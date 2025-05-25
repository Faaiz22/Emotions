# Emotion Classification Using Support Vector Machine (SVM)

A comprehensive machine learning pipeline for automated emotion classification from text data using Support Vector Machine algorithms with optimized performance and kernel comparison.

## Project Overview

This project implements an emotion classification system that can automatically predict emotions from text input (such as tweets) and determine which SVM kernel function provides the best classification performance. The system includes data preprocessing, feature extraction, model training with hyperparameter tuning, and comprehensive evaluation.

## Features

- **Multiple SVM Kernel Comparison**: Linear, RBF, Polynomial, and Sigmoid kernels
- **Automated Hyperparameter Tuning**: Grid search with cross-validation
- **Text Preprocessing Pipeline**: Cleaning, stopword removal, and lemmatization
- **TF-IDF Feature Extraction**: Optimized feature vectorization with n-grams
- **Performance Optimization**: Fast training with reduced dataset sampling
- **Comprehensive Evaluation**: Detailed classification reports and metrics
- **Model Persistence**: Save and load trained models and vectorizers

## File Structure

```
emotion-classification/
├── data_preprocessing.py      # Data preprocessing and feature extraction
├── model_training.py          # SVM model training and evaluation
├── main.py                   # Main pipeline execution
├── fast_emotion_model_*.pkl  # Trained SVM model
├── vectorizer_*.pkl          # TF-IDF vectorizer
└── README.md                # This file
```

## Dataset

The project uses an emotion dataset named emotions.csv from Kaggle but could not be uploaded because of it's size  and has  following structure: 
- **Text Column**: Raw text data for classification
- **Label Column**: Emotion labels (0-5 scale)

### Emotion Mapping
- 0: Neutral/Factual Statement
- 1: Mild Positive/Contentment
- 2: Slight Positivity/Nostalgia/Curiosity
- 3: Mild Negativity/Uncertainty/Stress
- 4: Strong Negative Emotion/Distress/Sadness
- 5: Extreme Distress/Overwhelming Emotion

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries
```bash
pip install pandas numpy scikit-learn nltk joblib openpyxl
```

### NLTK Data Download
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Usage

### Basic Usage

1. **Prepare your dataset**: Ensure your emotion dataset is in CSV or Excel format with 'text' and 'label' columns.

2. **Run the main pipeline**:
```python
python main.py
```

3. **Quick execution with optimizations**:
```python
# For fast results (2-5 minutes)
python fast_emotion_classification.py
```

### Custom Configuration

```python
from main import EmotionClassificationPipeline

# Initialize pipeline
pipeline = EmotionClassificationPipeline(
    csv_path="your_dataset.csv",
    text_column="text",
    label_column="label"
)

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Make predictions
emotion, label = pipeline.predict_emotion("I'm feeling great today!")
```

### Performance Optimization

For faster execution, adjust these parameters:

```python
# Ultra-fast (1-2 minutes)
SAMPLE_SIZE = 2000
MAX_FEATURES = 1000

# Balanced speed/accuracy (3-5 minutes)
SAMPLE_SIZE = 5000
MAX_FEATURES = 2000

# Better accuracy (5-10 minutes)
SAMPLE_SIZE = 10000
MAX_FEATURES = 3000
```

## Model Training Pipeline

### Step 1: Data Preprocessing
- Text cleaning and normalization
- Stopword removal
- Lemmatization
- TF-IDF feature extraction

### Step 2: Model Training
- Kernel performance comparison
- Grid search hyperparameter tuning
- Cross-validation evaluation

### Step 3: Model Evaluation
- Test set performance metrics
- Detailed classification reports
- Confusion matrix analysis

### Step 4: Model Persistence
- Save trained models
- Save feature vectorizers
- Export results and metrics

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average precision
- **Recall**: Per-class and weighted average recall
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation Scores**: K-fold validation results

## Kernel Comparison

The system automatically compares different SVM kernels:
- **Linear Kernel**: Fast, works well with high-dimensional data
- **RBF Kernel**: Good for non-linear relationships
- **Polynomial Kernel**: Captures polynomial relationships
- **Sigmoid Kernel**: Neural network-like activation

## Output Examples

### Kernel Performance Comparison
```
Linear kernel: 0.8250 (+/- 0.0150) - 2.5s
RBF kernel: 0.8180 (+/- 0.0200) - 15.2s
Polynomial kernel: 0.7950 (+/- 0.0180) - 8.7s
```

### Sample Predictions
```
'The weather is nice today' → Neutral/Factual Statement
'I'm feeling great about this project' → Mild Positive/Contentment
'I'm completely devastated by this news' → Strong Negative Emotion/Distress/Sadness
```

## Configuration Options

### Data Preprocessing
- `max_features`: Maximum TF-IDF features (default: 10000)
- `ngram_range`: N-gram range for features (default: (1,2))
- `min_df`: Minimum document frequency (default: 2)
- `max_df`: Maximum document frequency (default: 0.95)

### Model Training
- `cv_folds`: Cross-validation folds (default: 5)
- `test_size`: Test set proportion (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce sample size and max_features
2. **Slow Training**: Use the fast_emotion_classification script
3. **NLTK Data Missing**: Download required NLTK datasets
4. **File Not Found**: Ensure dataset path is correct

### Performance Tips

- Start with smaller datasets (2000-5000 samples)
- Use linear kernel for initial testing
- Reduce TF-IDF features for faster training
- Use 3-fold CV instead of 5-fold for speed


