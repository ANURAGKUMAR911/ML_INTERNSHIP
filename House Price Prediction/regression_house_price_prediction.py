import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
MODEL_DIR = 'saved_models'
DATA_FILE = 'BostonHousing.csv'
TARGET_COL = 'medv'

os.makedirs(MODEL_DIR, exist_ok=True)

class HousePricePredictor:
    def __init__(self):
        """Initialize the predictor with default models and scaler"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15, 
                random_state=RANDOM_STATE, 
                n_jobs=-1
            )
        }
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self, data_path=DATA_FILE):
        """Load and preprocess the dataset"""
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            
            # Rename target column for clarity
            df = df.rename(columns={TARGET_COL: 'Price'})
            
            # Separate features and target
            self.X = df.drop('Price', axis=1)
            self.y = df['Price']
            self.feature_names = self.X.columns.tolist()
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def preprocess_data(self, test_size=0.2):
        """Preprocess data and split into train/test sets"""
        # Handle missing values
        if self.X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.X = self.X.fillna(self.X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=RANDOM_STATE
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models with cross-validation"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='r2', n_jobs=-1
            )
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"  Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate models on test set"""
        results = {}
        
        for name, model_data in models.items():
            model = model_data['model']
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'y_pred': y_pred
            }
            
            # Update results
            results[name] = {**model_data, **metrics}
            
            # Print metrics
            print(f"\n{name} Test Performance:")
            print(f"  RMSE: ${metrics['rmse']:,.2f}")
            print(f"  MAE:  ${metrics['mae']:,.2f}")
            print(f"  R²:   {metrics['r2']:.4f}")
        
        return results
    
    def save_models(self, models, model_dir=MODEL_DIR):
        """Save trained models and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model_data in models.items():
            model_path = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
            joblib.dump(model_data['model'], model_path)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        # Save feature names
        joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        
        print(f"\nModels and scaler saved to {model_dir}/")
    
    def load_models(self, model_dir=MODEL_DIR):
        """Load trained models and scaler"""
        try:
            models = {}
            
            # Load models
            for name in self.models.keys():
                model_path = os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl")
                models[name] = {'model': joblib.load(model_path)}
            
            # Load scaler
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            
            # Load feature names
            self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
            
            return models
            
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)
    
    def make_predictions(self, models, X):
        """Make predictions using trained models"""
        # Ensure input has the same features as training data
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = {}
        for name, model_data in models.items():
            predictions[name] = model_data['model'].predict(X_scaled)
        
        return predictions
    
    def plot_results(self, y_true, results, save_path='house_price_prediction.png'):
        """Create and save visualization of results"""
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot each model's results
        for idx, (name, data) in enumerate(results.items()):
            y_pred = data['y_pred']
            residuals = y_true - y_pred
            
            # Actual vs Predicted
            ax = axes[0, idx]
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Price ($1000s)')
            ax.set_ylabel('Predicted Price ($1000s)')
            ax.set_title(f'{name} (R² = {data["r2"]:.4f})')
            ax.grid(True, alpha=0.3)
            
            # Residuals
            ax = axes[1, idx]
            ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Price ($1000s)')
            ax.set_ylabel('Residuals ($1000s)')
            ax.set_title(f'{name} - Residuals')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='House Price Prediction System')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    return parser.parse_args()

def main():
    """Main function to run the house price prediction system"""
    args = parse_arguments()
    predictor = HousePricePredictor()
    
    # If no arguments provided, run full pipeline
    if not any(vars(args).values()):
        args.train = True
        args.visualize = True
    
    # Load data
    print("Loading data...")
    df = predictor.load_data()
    
    if args.train:
        print("\n=== Training Models ===")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = predictor.preprocess_data()
        
        # Train models
        models = predictor.train_models(X_train, y_train)
        
        # Evaluate models
        print("\n=== Evaluating Models ===")
        results = predictor.evaluate_models(models, X_test, y_test)
        
        # Save models
        predictor.save_models(models)
        
        # Generate visualizations
        if args.visualize:
            print("\n=== Generating Visualizations ===")
            predictor.plot_results(y_test, results)
    
    if args.predict:
        print("\n=== Making Predictions ===")
        
        # Load models
        models = predictor.load_models()
        
        # Example prediction
        example_data = pd.DataFrame({
            'crim': [0.03, 0.1],
            'zn': [20, 0],
            'indus': [2.5, 8.0],
            'chas': [0, 0],
            'nox': [0.5, 0.6],
            'rm': [6.5, 6.0],
            'age': [50, 80],
            'dis': [4.0, 2.5],
            'rad': [1, 5],
            'tax': [300, 400],
            'ptratio': [15, 20],
            'b': [395, 390],
            'lstat': [5, 15]
        })
        
        predictions = predictor.make_predictions(models, example_data)
        
        print("\nExample Predictions:")
        print("-" * 50)
        for name, preds in predictions.items():
            print(f"\n{name}:")
            for i, pred in enumerate(preds, 1):
                print(f"  Property {i}: ${pred:,.2f}")

if __name__ == "__main__":
    main()
