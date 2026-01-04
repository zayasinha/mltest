import streamlit as st
import pandas as pd
import json
import os
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(page_title="ML Project Dashboard", page_icon="📊", layout="wide")

# Title
st.title("🚀 ML Project Dashboard")

# Load data
@st.cache_data
def load_model_report():
    with open('artifacts/model_report.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    with open('artifacts/feature_importance.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_raw_data():
    return pd.read_csv('artifacts/raw.csv')

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Comparison", "Feature Importance", "Data Exploration", "MLflow Runs"])

if page == "Overview":
    st.header("📈 Project Overview")

    # Load data
    model_report = load_model_report()
    raw_data = load_raw_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Models Trained", len(model_report))
        st.metric("Dataset Size", f"{len(raw_data):,}")

    with col2:
        # Find best model by test R2
        valid_models = [(model, data['metrics'].get('test_r2', float('-inf')))
                       for model, data in model_report.items()
                       if 'metrics' in data and 'test_r2' in data['metrics']]

        if valid_models:
            best_model = max(valid_models, key=lambda x: x[1])
            st.metric("Best Model", best_model[0])
            st.metric("Best Test R²", f"{best_model[1]:.3f}")
        else:
            st.metric("Best Model", "N/A")
            st.metric("Best Test R²", "N/A")

    with col3:
        st.metric("Features", len(raw_data.columns) - 1)  # Assuming last column is target

elif page == "Model Comparison":
    st.header("⚖️ Model Comparison")

    model_report = load_model_report()

    # Filter models with metrics
    valid_models = {k: v for k, v in model_report.items() if 'metrics' in v}

    if valid_models:
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_data in valid_models.items():
            metrics = model_data['metrics']
            comparison_data.append({
                'Model': model_name,
                'Train R²': metrics.get('train_r2', None),
                'Test R²': metrics.get('test_r2', None),
                'Train RMSE': metrics.get('train_rmse', None),
                'Test RMSE': metrics.get('test_rmse', None),
                'Train MAE': metrics.get('train_mae', None),
                'Test MAE': metrics.get('test_mae', None),
                'Overfitting Gap': metrics.get('overfitting_gap', None)
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Display table
        st.dataframe(df_comparison)

        # R² comparison chart
        fig_r2 = px.bar(df_comparison.dropna(subset=['Test R²']),
                       x='Model', y=['Train R²', 'Test R²'],
                       title="R² Scores Comparison",
                       barmode='group')
        st.plotly_chart(fig_r2)

        # RMSE comparison
        fig_rmse = px.bar(df_comparison.dropna(subset=['Test RMSE']),
                         x='Model', y=['Train RMSE', 'Test RMSE'],
                         title="RMSE Comparison",
                         barmode='group')
        st.plotly_chart(fig_rmse)

    else:
        st.warning("No valid model metrics found.")

elif page == "Feature Importance":
    st.header("🎯 Feature Importance")

    feature_imp = load_feature_importance()

    if feature_imp:
        # Sort features by importance
        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)

        # Create dataframe
        df_feat = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])

        # Plot top 20 features
        fig = px.bar(df_feat.head(20), x='Importance', y='Feature',
                    orientation='h', title="Top 20 Feature Importance")
        st.plotly_chart(fig)

        # Display table
        st.dataframe(df_feat)

    else:
        st.warning("Feature importance data not found.")

elif page == "Data Exploration":
    st.header("🔍 Data Exploration")

    raw_data = load_raw_data()

    st.subheader("Dataset Overview")
    st.write(f"Shape: {raw_data.shape}")
    st.write("Data Types:")
    st.write(raw_data.dtypes)

    st.subheader("Sample Data")
    st.dataframe(raw_data.head())

    st.subheader("Statistics")
    st.dataframe(raw_data.describe())

    # Correlation heatmap if numeric columns exist
    numeric_cols = raw_data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        st.subheader("Correlation Matrix")
        corr = raw_data[numeric_cols].corr()
        fig_corr = px.imshow(corr, title="Feature Correlation Matrix")
        st.plotly_chart(fig_corr)

elif page == "MLflow Runs":
    st.header("🏃 MLflow Experiment Runs")

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"file:{os.path.abspath('mlruns')}")

        # Get experiment
        experiment = mlflow.get_experiment("0")
        runs = mlflow.search_runs(experiment_ids=["0"])

        if not runs.empty:
            st.write(f"Total Runs: {len(runs)}")

            # Display runs table
            st.dataframe(runs[['run_id', 'start_time', 'status', 'metrics.mae', 'metrics.r2', 'metrics.rmse',
                             'params.learning_rate', 'params.n_estimators']].head(20))

            # Metrics over time if available
            if 'metrics.r2' in runs.columns:
                fig = px.scatter(runs, x='start_time', y='metrics.r2',
                               title="R² Score Over Time")
                st.plotly_chart(fig)
        else:
            st.info("No MLflow runs found.")

    except Exception as e:
        st.error(f"Error loading MLflow data: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")