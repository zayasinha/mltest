# BMW Car Price Prediction

## Project Overview
Predicts BMW car prices using machine learning and provides interactive dashboards for data exploration and model performance analysis.

## Dataset
- 10,783 BMW cars (2010-2020)
- Features: model, year, transmission, mileage, fuelType, tax, mpg, engineSize
- Target: price (continuous)

## Tools & Technologies
- Python 3.8+
- pandas, numpy
- scikit-learn, catboost, xgboost
- Streamlit, Plotly
- MLflow

## Analysis Performed
- Data preprocessing and feature engineering
- Model training with 7 algorithms (CatBoost, Random Forest, XGBoost, etc.)
- Hyperparameter tuning with cross-validation
- Feature importance analysis
- Interactive dashboards for data exploration and model comparison

## Key Insights / Results
- Best model: CatBoost Regressor (R² = 0.945)
- Top price drivers: engine size (28%), year (25%), mileage (19%)
- Average car price: £22,500
- Market composition: 52% diesel, 45% petrol
- Price range: £495 - £159,999

## How to Run
```bash
pip install -r requirements.txt
streamlit run data_exploration_dashboard.py  # Data dashboard
streamlit run ml_dashboard_deploy.py         # ML dashboard
```