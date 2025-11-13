import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
print("Loading dataset.")
data = pd.read_csv('BostonHousing.csv')

# Data Exploration
print("\n Dataset Information")
print(f"Shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nMissing values in each column:")
print(data.isnull().sum())
print("\nStatistical summary:")
print(data.describe())

# Preprocessing
print("\n Preprocessing Data")
X = data.drop('medv', axis=1)
y = data['medv']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Helper function to train and evaluate a model"""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    # Print results
    print(f"\n{model_name} Results")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Cross-validated R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': data.drop('medv', axis=1).columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.savefig('feature_importance.png', bbox_inches='tight')
            plt.close()
            print("Saved feature importance plot: feature_importance.png")
        except Exception as e:
            print(f"Warning: Could not generate feature importance plot - {str(e)}")
    
    # For linear regression coefficients
    elif hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': data.drop('medv', axis=1).columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        print("\nLinear Regression Coefficients:")
        print(coef_df)
    
    try:
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices ($1000s)')
        plt.ylabel('Predicted Prices ($1000s)')
        plt.title(f'{model_name} - Actual vs. Predicted Prices')
        pred_plot_path = f'{model_name.lower().replace(" ", "_")}_predictions.png'
        plt.savefig(pred_plot_path, bbox_inches='tight')
        plt.close()
        print(f"  - Saved predictions plot: {pred_plot_path}")
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residuals Plot')
        res_plot_path = f'{model_name.lower().replace(" ", "_")}_residuals.png'
        plt.savefig(res_plot_path, bbox_inches='tight')
        plt.close()
        print(f"  - Saved residuals plot: {res_plot_path}")
        
    except Exception as e:
        print(f"  - Warning: Could not generate plots - {str(e)}")
        pass
    
    return model, y_pred

# Initialize models with simpler configurations
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(
        n_estimators=50,  # Reduced number of estimators
        max_depth=5,      # Limit tree depth
        random_state=42,
        n_jobs=1          # Use single core to avoid threading issues
    )
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training and evaluating {name}...")
    trained_model, y_pred = evaluate_model(
        model, X_train, X_test, y_train, y_test, name
    )
    results[name] = {
        'model': trained_model,
        'predictions': y_pred,
        'actual': y_test
    }

# Save the best model
best_model_name = max(results, key=lambda k: r2_score(results[k]['actual'], results[k]['predictions']))
best_model = results[best_model_name]['model']
best_r2 = r2_score(results[best_model_name]['actual'], results[best_model_name]['predictions'])

print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {best_r2:.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(results[best_model_name]['actual'], results[best_model_name]['predictions'])):.4f}")
print(f"MAE: {mean_absolute_error(results[best_model_name]['actual'], results[best_model_name]['predictions']):.4f}")

# Save model and scaler
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel saved successfully.")
