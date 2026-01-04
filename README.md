# ML Project Dashboard

This is a machine learning project with automated data ingestion, transformation, model training, and evaluation pipeline.

## Features

- **Data Ingestion**: Automated data loading and preprocessing
- **Model Training**: Multiple algorithms (Linear Regression, Random Forest, XGBoost, CatBoost, etc.)
- **Model Evaluation**: Comprehensive metrics and cross-validation
- **MLflow Integration**: Experiment tracking and model versioning
- **Interactive Dashboard**: Visualize results and insights

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Pipeline
Run the main training pipeline:
```bash
python app.py
```

### Dashboard
Launch the interactive dashboard:
```bash
python run_dashboard.py
```

Or directly with Streamlit:
```bash
streamlit run dashboard.py
```

The dashboard includes:
- **Overview**: Project summary and key metrics
- **Model Comparison**: Compare performance across all trained models
- **Feature Importance**: Visualize which features contribute most to predictions
- **Data Exploration**: Explore the dataset with statistics and correlations
- **MLflow Runs**: View experiment tracking data

## Project Structure

```
├── app.py                 # Main training pipeline
├── dashboard.py           # Streamlit dashboard
├── run_dashboard.py       # Dashboard runner script
├── requirements.txt       # Python dependencies
├── artifacts/            # Model artifacts and data (not in repo)
│   ├── feature_importance.json
│   ├── model_report.json
│   └── ... (generated files)
├── mlruns/              # MLflow experiment runs (not in repo)
├── src/mltest/          # Source code
│   ├── components/      # Pipeline components
│   └── pipelines/       # Pipeline orchestrators
├── notebook/            # Jupyter notebooks
└── logs/                # Training logs (not in repo)
```

## Deployment

### Local Development
1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`
5. Run training: `python app.py`
6. Launch dashboard: `python run_dashboard.py`

### GitHub Repository
- **Source code**: ✅ Included
- **Models & artifacts**: ❌ Excluded (regenerate with `python app.py`)
- **MLflow runs**: ❌ Excluded (regenerate during training)
- **Data files**: ❌ Excluded (add your own data to `artifacts/raw.csv`)

## MLflow Setup

To enable experiment tracking with DagsHub:

```python
import dagshub
dagshub.init(repo_owner='zayasinha', repo_name='mltest', mlflow=True)

import mlflow
with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
```