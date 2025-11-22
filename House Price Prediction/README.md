# House Price Prediction

## Project Files
1. `ml_house_price_prediction.py` - Comprehensive ML approach with multiple algorithms
2. `regression_house_price_prediction.py` - Focused on linear regression implementation

## Installation
```bash
pip install -r requirements.txt
```

## Running the Models

### 1. ML Approach (Comprehensive)
```bash
# Train models and generate visualizations
python ml_house_price_prediction.py --train --visualize

# Make predictions
python ml_house_price_prediction.py --predict
```

### 2. Regression Approach (Focused)
```bash
# Run the regression model
python regression_house_price_prediction.py
```

## Output Files
- `saved_models/` - Directory containing trained models
- `*.png` - Visualization files including:
  - Actual vs Predicted prices
  - Residual plots
  - Feature importance

## Requirements
- Python 3.6+
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib


