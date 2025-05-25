import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import joblib
import time

class EmotionClassifier:
    def __init__(self):
        """Initialize the emotion classifier."""
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.cv_results = {}
        
    def train_svm_with_grid_search(self, X_train, y_train, cv_folds=5):
        """
        Train SVM classifier with grid search for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Training results including best parameters and scores
        """
        print("Starting SVM training with Grid Search...")
        
        # Define parameter grid for different kernels
        param_grid = [
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10, 100]
            },
            {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            {
                'kernel': ['poly'],
                'C': [0.1, 1, 10],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto']
            }
        ]
        
        # Initialize SVM classifier
        svm = SVC(random_state=42)
        
        # Perform grid search with cross-validation
        print(f"Performing grid search with {cv_folds}-fold cross-validation...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,  # Use all available processors
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Store the best model and parameters
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # Store detailed results
        results = {
            'best_params': self.best_params,
            'best_cv_score': self.best_score,
            'training_time': training_time,
            'total_combinations': len(grid_search.cv_results_['params'])
        }
        
        print(f"\nGrid Search completed in {training_time:.2f} seconds!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return results
    
    def evaluate_kernel_performance(self, X_train, y_train, cv_folds=5):
        """
        Evaluate different SVM kernels separately for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            dict: Performance comparison of different kernels
        """
        print("Evaluating different SVM kernels...")
        
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        kernel_results = {}
        
        for kernel in kernels:
            print(f"Evaluating {kernel} kernel...")
            
            # Create SVM with current kernel
            svm = SVC(kernel=kernel, random_state=42)
            
            # Perform cross-validation
            start_time = time.time()
            cv_scores = cross_val_score(svm, X_train, y_train, cv=cv_folds, scoring='accuracy')
            end_time = time.time()
            
            kernel_results[kernel] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores,
                'training_time': end_time - start_time
            }
            
            print(f"{kernel} kernel: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return kernel_results
    
    def evaluate_model(self, X_test, y_test, target_names=None):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: List of class names for better reporting
            
        Returns:
            dict: Comprehensive evaluation results
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
        
        print("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return results
    
    def print_detailed_classification_report(self, evaluation_results, target_names=None):
        """
        Print a detailed classification report.
        
        Args:
            evaluation_results: Results from evaluate_model method
            target_names: List of class names
        """
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        
        if target_names:
            print("\nPer-class Performance:")
            for i, class_name in enumerate(target_names):
                if str(i) in evaluation_results['classification_report']:
                    metrics = evaluation_results['classification_report'][str(i)]
                    print(f"{class_name}:")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  F1-Score: {metrics['f1-score']:.4f}")
                    print(f"  Support: {metrics['support']}")
                    print()
        
        print("Overall Performance:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Macro Avg Precision: {evaluation_results['classification_report']['macro avg']['precision']:.4f}")
        print(f"Macro Avg Recall: {evaluation_results['classification_report']['macro avg']['recall']:.4f}")
        print(f"Macro Avg F1-Score: {evaluation_results['classification_report']['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg Precision: {evaluation_results['classification_report']['weighted avg']['precision']:.4f}")
        print(f"Weighted Avg Recall: {evaluation_results['classification_report']['weighted avg']['recall']:.4f}")
        print(f"Weighted Avg F1-Score: {evaluation_results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
        
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Predictions
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities (requires probability=True in SVM).
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Prediction probabilities
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Check if the model supports probability prediction
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            print("Warning: Current model doesn't support probability prediction.")
            print("Retrain with probability=True for probability estimates.")
            return None