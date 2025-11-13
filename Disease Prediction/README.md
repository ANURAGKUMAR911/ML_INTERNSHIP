# Disease Prediction System

## Quick Start

1. Install requirements:
```bash
pip install pandas scikit-learn matplotlib seaborn xgboost joblib

or 

pip install -r requirements.txt

2. Train a model:
```bash
python disease_prediction.py train heart
python disease_prediction.py train diabetes
python disease_prediction.py train parkinsons
```

3. Make predictions:
```bash
python disease_prediction.py predict heart
python disease_prediction.py predict diabetes
python disease_prediction.py predict parkinsons
```

4. List saved models:
```bash
python disease_prediction.py list
```

