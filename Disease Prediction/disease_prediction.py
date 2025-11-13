import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any, Optional, Union

# Constants
DISEASES = {
    'heart': {
        'name': 'Heart Disease',
        'data_url': 'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv',
        'target': 'target',
        'positive_class': 'Disease',
        'negative_class': 'No Disease'
    },
    'diabetes': {
        'name': 'Diabetes',
        'data_url': 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',
        'target': 'Outcome',
        'positive_class': 'Diabetic',
        'negative_class': 'Non-Diabetic'
    },
    'parkinsons': {
        'name': 'Parkinson\'s',
        'data_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
        'target': 'status',
        'positive_class': 'Has Parkinson\'s',
        'negative_class': 'Healthy'
    }
}

MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)  # Set default figure size

def load_and_preprocess_data(disease_type: str) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Load and preprocess the dataset for the specified disease.
    
    Args:
        disease_type: Type of disease ('heart', 'diabetes', or 'parkinsons')
        
    Returns:
        Tuple of (features, target, metadata)
    """
    disease_info = DISEASES[disease_type]
    print(f"\n{'='*50}")
    print(f"Loading {disease_info['name']} dataset...")
    
    try:
        # Load dataset
        data = pd.read_csv(disease_info['data_url'])
        
        # Special handling for Parkinson's dataset
        if disease_type == 'parkinsons':
            data = data.drop(['name'], axis=1)  # Drop name column
        
        # Basic data info
        print("\n=== Dataset Information ===")
        print(f"Shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Check for missing values
        print("\nMissing values in each column:")
        print(data.isnull().sum())
        
        # Separate features and target
        X = data.drop(disease_info['target'], axis=1)
        y = data[disease_info['target']]
        
        # For diabetes dataset, rename target values
        if disease_type == 'diabetes':
            y = y.map({0: 0, 1: 1})  # Ensure binary classification
        
        return X, y, disease_info
        
    except Exception as e:
        print(f"Error loading {disease_info['name']} dataset: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, model_name: str, disease_info: dict) -> dict:
    """
    Evaluate the model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Name of the model
        disease_info: Dictionary containing disease information
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities are available
        if hasattr(model, 'predict_proba'):
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_test, 
            y_pred, 
            target_names=[disease_info['negative_class'], disease_info['positive_class']],
            output_dict=True
        )
        
        # Create visualization directory
        os.makedirs('evaluation_plots', exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[disease_info['negative_class'], disease_info['positive_class']],
            yticklabels=[disease_info['negative_class'], disease_info['positive_class']]
        )
        plt.title(f"{disease_info['name']} - {model_name}\nConfusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"evaluation_plots/{disease_info['name'].lower().replace(' ', '_')}_cm_{model_name.lower().replace(' ', '_')}.png")
        plt.close()
        
        # Plot ROC curve if probabilities are available
        if 'roc_auc' in metrics:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f"ROC curve (AUC = {metrics['roc_auc']:.4f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"{disease_info['name']} - {model_name}\nROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"evaluation_plots/{disease_info['name'].lower().replace(' ', '_')}_roc_{model_name.lower().replace(' ', '_')}.png")
            plt.close()
            
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model {model_name}: {str(e)}")
        raise

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    elif hasattr(model, 'coef_'):
        coef = model.coef_[0]
        indices = np.argsort(np.abs(coef))[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Coefficients - {model_name}')
        plt.bar(range(len(coef)), coef[indices])
        plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'feature_coefficients_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def train_and_evaluate_models(disease_type: str) -> dict:
    """
    Train and evaluate models for a specific disease.
    
    Args:
        disease_type: Type of disease ('heart', 'diabetes', or 'parkinsons')
        
    Returns:
        Dictionary containing training results
    """
    try:
        # Load and preprocess data
        X, y, disease_info = load_and_preprocess_data(disease_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to evaluate
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced'
            ),
            'SVM': SVC(
                probability=True, 
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                scale_pos_weight=sum(y==0)/sum(y==1)  # Handle class imbalance
            )
        }
        
        # Train and evaluate models
        results = []
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training {name} for {disease_info['name']}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                metrics = evaluate_model(model, X_test_scaled, y_test, name, disease_info)
                results.append(metrics)
                
                # Print metrics
                print(f"\n{name} Performance:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
                
                # Save the trained model
                model_dir = os.path.join(MODEL_DIR, disease_type)
                os.makedirs(model_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}_{timestamp}.pkl")
                joblib.dump({
                    'model': model,
                    'scaler': scaler,
                    'metrics': metrics,
                    'feature_names': X.columns.tolist(),
                    'timestamp': timestamp,
                    'disease_type': disease_type
                }, model_path)
                
                print(f"\nModel saved to: {model_path}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No models were successfully trained.")
        
        # Find best model based on ROC-AUC or F1 score
        best_metric = 'roc_auc' if 'roc_auc' in results[0] else 'f1'
        best_model_info = max(results, key=lambda x: x[best_metric])
        
        print("\n" + "="*50)
        print(f"Best Model for {disease_info['name']}: {best_model_info['model']}")
        print(f"Best {best_metric.upper()}: {best_model_info[best_metric]:.4f}")
        
        return {
            'disease': disease_info['name'],
            'best_model': best_model_info['model'],
            'best_metric': best_metric,
            'best_score': best_model_info[best_metric],
            'results': results
        }
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise

def predict_disease(disease_type: str, input_data: dict) -> dict:
    """
    Make a prediction using a trained model.
    
    Args:
        disease_type: Type of disease ('heart', 'diabetes', or 'parkinsons')
        input_data: Dictionary of input features
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        model_dir = os.path.join(MODEL_DIR, disease_type)
        if not os.path.exists(model_dir):
            return {"error": f"No models found for {disease_type}. Please train a model first."}
        
        # Get the most recent model
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            return {"error": f"No model files found for {disease_type}."}
            
        latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        model_path = os.path.join(model_dir, latest_model)
        
        # Load the model
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data.get('feature_names', [])
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            return {"error": f"Missing required features: {', '.join(missing_features)}"}
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # Probability of positive class
        
        disease_info = DISEASES.get(disease_type, {
            'positive_class': 'Positive',
            'negative_class': 'Negative'
        })
        
        return {
            'disease': disease_info.get('name', disease_type),
            'prediction': int(prediction),
            'prediction_label': disease_info['positive_class'] if prediction == 1 else disease_info['negative_class'],
            'probability': float(probability),
            'model': type(model).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Disease Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('disease', choices=DISEASES.keys(), help='Type of disease to predict')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a prediction')
    predict_parser.add_argument('disease', choices=DISEASES.keys(), help='Type of disease to predict')
    
    # List models command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Train a new model
        train_and_evaluate_models(args.disease)
        
    elif args.command == 'predict':
        # Example prediction (in a real app, you would collect input from a form or API)
        print(f"\nEnter the following features for {DISEASES[args.disease]['name']} prediction:")
        
        # This is a simplified example - in a real app, you would collect input properly
        example_input = {}
        if args.disease == 'heart':
            example_input = {
                'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212,
                'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0, 'oldpeak': 1.0,
                'slope': 2, 'ca': 2, 'thal': 3
            }
        elif args.disease == 'diabetes':
            example_input = {
                'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35,
                'Insulin': 0, 'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50
            }
        elif args.disease == 'parkinsons':
            example_input = {
                'MDVP:Fo(Hz)': 119.992, 'MDVP:Fhi(Hz)': 157.302, 'MDVP:Flo(Hz)': 74.997,
                'MDVP:Jitter(%)': 0.00784, 'MDVP:Jitter(Abs)': 0.00007, 'MDVP:RAP': 0.0037,
                'MDVP:PPQ': 0.00554, 'Jitter:DDP': 0.01109, 'MDVP:Shimmer': 0.04374,
                'MDVP:Shimmer(dB)': 0.426, 'Shimmer:APQ3': 0.02182, 'Shimmer:APQ5': 0.0313,
                'MDVP:APQ': 0.02971, 'Shimmer:DDA': 0.06545, 'NHR': 0.02211, 'HNR': 21.033,
                'RPDE': 0.414783, 'DFA': 0.815285, 'spread1': -4.813031, 'spread2': 0.266482,
                'D2': 2.301442, 'PPE': 0.284654
            }
            
        print("\nExample input features:")
        for k, v in example_input.items():
            print(f"{k}: {v}")
            
        result = predict_disease(args.disease, example_input)
        print("\nPrediction Result:")
        print(f"Disease: {result['disease']}")
        print(f"Prediction: {result['prediction_label']} (Probability: {result['probability']:.4f})")
        print(f"Model: {result['model']}")
        
    elif args.command == 'list':
        # List available models
        print("\nAvailable disease models:")
        for disease in DISEASES:
            model_dir = os.path.join(MODEL_DIR, disease)
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                print(f"\n{DISEASES[disease]['name']}:")
                if models:
                    latest = max(models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
                    print(f"  - Latest model: {latest}")
                    print(f"  - Total models: {len(models)}")
                else:
                    print("  No models trained yet.")
            else:
                print(f"\n{DISEASES[disease]['name']}: No models trained yet.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
