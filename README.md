# BMW Car Price Prediction - MLTest

A production-ready machine learning pipeline for predicting BMW car prices using multiple regression algorithms with comprehensive data preprocessing and model evaluation.

## 📋 Project Overview

This project demonstrates a complete ML lifecycle including:
- **Data Ingestion**: Reading from MySQL database and splitting into train/test sets
- **Data Transformation**: Feature engineering, scaling, and encoding
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and best model selection
- **Prediction Pipeline**: Make predictions on new data

## 🏗️ Project Structure

```
mltest/
├── src/mltest/
│   ├── components/
│   │   ├── data_ingestion.py       # Read data from database
│   │   ├── data_transformation.py  # Preprocess & feature engineering
│   │   └── model_tranier.py        # Train & evaluate models
│   ├── pipelines/
│   │   ├── training_pipeline.py    # Orchestrate training workflow
│   │   └── prediction_pipeline.py  # Make predictions on new data
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Helper functions
├── artifacts/                      # Saved models & preprocessor
├── logs/                          # Execution logs
├── notebook/                      # Jupyter notebooks for EDA
├── app.py                         # Main entry point
├── requirements.txt               # Package dependencies
├── setup.py                       # Package configuration
└── README.md                      # This file
```

## 📊 Dataset

**BMW Car Dataset** with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| model | Categorical | Car model name |
| year | Numerical | Year of manufacture |
| price | Target | Car price (£) |
| transmission | Categorical | Manual/Automatic/Semi-Auto |
| mileage | Numerical | Miles driven |
| fuelType | Categorical | Petrol/Diesel |
| tax | Numerical | Annual tax (£) |
| mpg | Numerical | Miles per gallon |
| engineSize | Numerical | Engine size (L) |

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- MySQL database with BMW data
- Conda or virtual environment

### Setup

1. **Clone the repository**
   ```bash
   cd c:\apple
   ```

2. **Create and activate virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure database credentials** (create `.env` file)
   ```
   host=localhost
   user=your_username
   password=your_password
   db=your_database
   ```

## 🚀 Usage

### Run the Complete Training Pipeline

```bash
python app.py
```

This will:
1. Ingest data from MySQL
2. Split into train/test sets (80/20)
3. Transform and preprocess features
4. Train 9 different regression models
5. Evaluate and select the best model
6. Save model and preprocessor artifacts

### Make Predictions

```python
from src.mltest.pipelines.prediction_pipeline import PredictionPipeline

# Initialize predictor
predictor = PredictionPipeline()

# Define car features
car = {
    'model': '3 Series',
    'year': 2020,
    'transmission': 'Semi-Auto',
    'mileage': 5000,
    'fuelType': 'Diesel',
    'tax': 145,
    'mpg': 52.3,
    'engineSize': 2.0
}

# Get prediction
predicted_price = predictor.predict(car)
print(f"Predicted price: £{predicted_price:.2f}")
```

## 📈 Models Trained

The pipeline trains and evaluates 9 regression models:

1. **Linear Regression** - Baseline linear model
2. **Decision Tree** - Tree-based non-linear model
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential boosting approach
5. **XGBoost** - Extreme gradient boosting
6. **CatBoost** - Categorical boosting
7. **AdaBoost** - Adaptive boosting
8. **KNN** - K-Nearest Neighbors
9. **Bagging** - Bootstrap aggregating

The best model is selected based on R² score and saved for predictions.

## 📊 Evaluation Metrics

- **R² Score** - Coefficient of determination (0-1, higher is better)
- **RMSE** - Root Mean Squared Error (lower is better)
- **MAE** - Mean Absolute Error (lower is better)
- **MSE** - Mean Squared Error (lower is better)

## 🔄 Data Preprocessing

### Numerical Features
- **Scaling**: StandardScaler (mean=0, std=1)
- **Features**: year, mileage, engineSize, tax, mpg

### Categorical Features
- **Encoding**: One-Hot Encoding
- **Features**: model, transmission, fuelType
- **Scaling**: StandardScaler (with_mean=False for sparse data)

## 📝 Logging

All pipeline execution is logged to timestamped log files in the `logs/` directory with format:
```
[TIMESTAMP] LINE_NO MODULE - LEVEL - MESSAGE
```

## 🛡️ Error Handling

Custom exception handling provides detailed error information including:
- File name where error occurred
- Line number
- Error message
- Full traceback in logs

## 🔧 Configuration

### Model Trainer Config
- **Output**: `artifacts/model.pkl`

### Data Transformation Config
- **Output**: `artifacts/preprocessor.pkl`

### Data Ingestion Config
- **Train**: `artifacts/train.csv` (80%)
- **Test**: `artifacts/test.csv` (20%)
- **Raw**: `artifacts/raw.csv`

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):
- `numpy==2.3.4` - Numerical computing
- `pandas>=2.0` - Data manipulation
- `scikit-learn` - ML algorithms
- `xgboost` - XGBoost models
- `catboost` - CatBoost models
- `mlflow>=2.10.0` - Experiment tracking
- `flask` - Web framework
- `dvc` - Data version control
- `pymysql` - MySQL connection
- `python-dotenv` - Environment variables

## 🐳 Docker Support

Build and run with Docker:
```bash
docker build -t mltest .
docker run mltest
```

## 📌 Notes

- Data is sourced from a MySQL database (BMW car listings)
- The pipeline is modular and can be extended with new models
- All outputs (models, preprocessor, logs) are versioned and tracked
- Use DVC for data version control

## 👤 Author

**Jaya Sinha** - sinhajaya601@gmail.com

## 📄 License

This project is part of the MLTest learning series.

---

**Status**: ✅ Production Ready | **Last Updated**: November 2025