import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="BMW Cars Data Exploration Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚗 BMW Cars Data Exploration Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    """Load and cache the BMW car dataset"""
    try:
        # Try multiple possible paths for deployment compatibility
        possible_paths = [
            'notebook/data/raw.csv',
            'data/raw.csv',
            'raw.csv'
        ]

        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                return df
            except FileNotFoundError:
                continue

        # If no file found, create sample data for demo
        st.warning("⚠️ Dataset not found. Using sample data for demonstration.")
        np.random.seed(42)
        n_samples = 10783  # Match original dataset size

        models = ['3 Series', '1 Series', '5 Series', 'X3', 'X5', '2 Series', '4 Series', 'X1', 'X4', '6 Series', '7 Series', 'Z4', 'M3', 'M4', 'M5', 'M6', 'i3', 'i8']
        fuel_types = ['Diesel', 'Petrol', 'Hybrid', 'Electric']
        transmissions = ['Manual', 'Automatic', 'Semi-Auto']

        data = {
            'model': np.random.choice(models, n_samples, p=[0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]),
            'year': np.random.randint(2010, 2021, n_samples),
            'price': np.random.normal(25000, 15000, n_samples).clip(3000, 150000),
            'mileage': np.random.normal(45000, 30000, n_samples).clip(1000, 200000),
            'fuelType': np.random.choice(fuel_types, n_samples, p=[0.45, 0.35, 0.15, 0.05]),
            'transmission': np.random.choice(transmissions, n_samples, p=[0.4, 0.4, 0.2]),
            'engineSize': np.random.choice([1.5, 2.0, 3.0, 4.0, 4.4, 6.0], n_samples, p=[0.1, 0.4, 0.3, 0.1, 0.05, 0.05]),
            'tax': np.random.normal(180, 80, n_samples).clip(0, 580),
            'mpg': np.random.normal(45, 20, n_samples).clip(10, 150)
        }

        df = pd.DataFrame(data)
        df['price'] = df['price'].astype(int)
        df['mileage'] = df['mileage'].astype(int)
        df['tax'] = df['tax'].astype(int)
        df['mpg'] = df['mpg'].round(1)

        return df

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# Load data
df = load_data()
if df is None:
    st.stop()

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Explore:",
    ["📊 Overview", "📈 Price Analysis", "🚙 Model Insights", "⛽ Fuel & Transmission",
     "📅 Year Trends", "🔗 Correlations", "📦 Distributions", "🎯 Outliers", "💡 Insights"]
)

# Color scheme for consistency
colors = px.colors.qualitative.Set3

# Main content based on selected page
if page == "📊 Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cars", f"{len(df):,}")

    with col2:
        st.metric("Unique Models", df['model'].nunique())

    with col3:
        st.metric("Price Range", f"£{df['price'].min():,} - £{df['price'].max():,}")

    with col4:
        st.metric("Avg Price", f"£{df['price'].mean():,.0f}")

    st.markdown("---")

    # Dataset info
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Dataset Information")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # Data types
        st.write("**Data Types:**")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"- {dtype}: {count} columns")

    with col2:
        st.subheader("🔍 Quick Stats")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Missing values
        missing = df.isnull().sum().sum()
        st.write(f"**Missing Values:** {missing}")

        # Duplicates
        duplicates = df.duplicated().sum()
        st.write(f"**Duplicate Rows:** {duplicates}")

        # Outliers (simple IQR method for price)
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR))).sum()
        st.write(f"**Price Outliers:** {outliers}")

    st.markdown("---")

    # Sample data preview
    st.subheader("👀 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

elif page == "📈 Price Analysis":
    st.header("📈 Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig = px.histogram(df, x='price', nbins=50,
                          title="Price Distribution",
                          color_discrete_sequence=[colors[0]])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Price box plot
        fig = px.box(df, y='price', title="Price Box Plot",
                    color_discrete_sequence=[colors[1]])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price by categories
    col1, col2 = st.columns(2)

    with col1:
        # Price by fuel type
        fig = px.box(df, x='fuelType', y='price',
                    title="Price by Fuel Type",
                    color='fuelType', color_discrete_sequence=colors)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Price by transmission
        fig = px.box(df, x='transmission', y='price',
                    title="Price by Transmission",
                    color='transmission', color_discrete_sequence=colors)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Price vs numeric features
    st.subheader("📊 Price vs Numeric Features")

    numeric_features = ['year', 'mileage', 'engineSize', 'tax', 'mpg']

    for i in range(0, len(numeric_features), 2):
        col1, col2 = st.columns(2)

        for j, col in enumerate([col1, col2]):
            if i + j < len(numeric_features):
                feature = numeric_features[i + j]
                with col:
                    fig = px.scatter(df, x=feature, y='price',
                                   title=f"Price vs {feature.title()}",
                                   color_discrete_sequence=[colors[(i+j) % len(colors)]],
                                   opacity=0.6)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

elif page == "🚙 Model Insights":
    st.header("🚙 Model Insights")

    # Model distribution
    col1, col2 = st.columns(2)

    with col1:
        model_counts = df['model'].value_counts()
        fig = px.bar(model_counts.head(10), x=model_counts.head(10).index,
                    y=model_counts.head(10).values,
                    title="Top 10 BMW Models",
                    color_discrete_sequence=[colors[0]])
        fig.update_layout(xaxis_title="Model", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Average price by model
        avg_price_by_model = df.groupby('model')['price'].mean().sort_values(ascending=False)
        fig = px.bar(avg_price_by_model.head(10),
                    title="Average Price by Model (Top 10)",
                    color_discrete_sequence=[colors[1]])
        fig.update_layout(xaxis_title="Model", yaxis_title="Average Price (£)", showlegend=False)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model performance analysis
    st.subheader("📊 Model Performance Analysis")

    model_stats = df.groupby('model').agg({
        'price': ['count', 'mean', 'std', 'min', 'max'],
        'mileage': 'mean',
        'year': 'mean'
    }).round(2)

    model_stats.columns = ['Count', 'Avg Price', 'Price Std', 'Min Price', 'Max Price', 'Avg Mileage', 'Avg Year']
    model_stats = model_stats.sort_values('Count', ascending=False)

    st.dataframe(model_stats.head(15), use_container_width=True)

    # Price range by model
    fig = px.scatter(df, x='model', y='price', color='model',
                    title="Price Distribution by Model",
                    color_discrete_sequence=colors)
    fig.update_layout(xaxis_title="Model", yaxis_title="Price (£)", showlegend=False)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

elif page == "⛽ Fuel & Transmission":
    st.header("⛽ Fuel & Transmission Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Fuel type distribution
        fuel_counts = df['fuelType'].value_counts()
        fig = px.pie(fuel_counts, values=fuel_counts.values, names=fuel_counts.index,
                    title="Fuel Type Distribution",
                    color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Transmission distribution
        trans_counts = df['transmission'].value_counts()
        fig = px.pie(trans_counts, values=trans_counts.values, names=trans_counts.index,
                    title="Transmission Distribution",
                    color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Fuel vs Transmission analysis
    st.subheader("🔄 Fuel Type vs Transmission")

    fuel_trans = pd.crosstab(df['fuelType'], df['transmission'])
    fig = px.imshow(fuel_trans, text_auto=True,
                   title="Fuel Type vs Transmission Heatmap",
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    # Average price by fuel and transmission
    col1, col2 = st.columns(2)

    with col1:
        avg_price_fuel = df.groupby('fuelType')['price'].mean().sort_values(ascending=False)
        fig = px.bar(avg_price_fuel, title="Average Price by Fuel Type",
                    color_discrete_sequence=colors)
        fig.update_layout(xaxis_title="Fuel Type", yaxis_title="Average Price (£)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_price_trans = df.groupby('transmission')['price'].mean().sort_values(ascending=False)
        fig = px.bar(avg_price_trans, title="Average Price by Transmission",
                    color_discrete_sequence=colors)
        fig.update_layout(xaxis_title="Transmission", yaxis_title="Average Price (£)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📅 Year Trends":
    st.header("📅 Year Trends")

    # Cars by year
    year_counts = df['year'].value_counts().sort_index()
    fig = px.bar(year_counts, x=year_counts.index, y=year_counts.values,
                title="Number of Cars by Year",
                color_discrete_sequence=[colors[0]])
    fig.update_layout(xaxis_title="Year", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price trends by year
    col1, col2 = st.columns(2)

    with col1:
        avg_price_year = df.groupby('year')['price'].mean()
        fig = px.line(avg_price_year, markers=True,
                     title="Average Price Trend by Year",
                     color_discrete_sequence=[colors[1]])
        fig.update_layout(xaxis_title="Year", yaxis_title="Average Price (£)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        median_price_year = df.groupby('year')['price'].median()
        fig = px.line(median_price_year, markers=True,
                     title="Median Price Trend by Year",
                     color_discrete_sequence=[colors[2]])
        fig.update_layout(xaxis_title="Year", yaxis_title="Median Price (£)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Year vs other features
    st.subheader("📊 Year vs Other Features")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(df, x='year', y='mileage', color='price',
                        title="Year vs Mileage (Color: Price)",
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='year', y='engineSize', color='price',
                        title="Year vs Engine Size (Color: Price)",
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔗 Correlations":
    st.header("🔗 Feature Correlations")

    # Correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig = px.imshow(corr_matrix, text_auto='.2f',
                   title="Correlation Matrix",
                   color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Strong correlations
    st.subheader("🎯 Strong Correlations")

    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))

    # Find strong correlations
    strong_corr = []
    for i in range(len(upper.columns)):
        for j in range(i+1, len(upper.columns)):
            corr_val = upper.iloc[i, j]
            if abs(corr_val) > 0.3:  # Threshold for strong correlation
                strong_corr.append({
                    'Feature 1': upper.columns[i],
                    'Feature 2': upper.columns[j],
                    'Correlation': corr_val
                })

    if strong_corr:
        corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(corr_df.round(3), use_container_width=True)

        # Visualize top correlations
        top_corr = corr_df.head(6)
        for _, row in top_corr.iterrows():
            fig = px.scatter(df, x=row['Feature 1'], y=row['Feature 2'],
                           title=f"{row['Feature 1'].title()} vs {row['Feature 2'].title()} (r = {row['Correlation']:.3f})",
                           color_discrete_sequence=[colors[len(strong_corr) % len(colors)]],
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No strong correlations found with the current threshold (>0.3).")

elif page == "📦 Distributions":
    st.header("📦 Feature Distributions")

    # Select feature to analyze
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_feature = st.selectbox("Select Feature to Analyze:", numeric_cols)

    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig = px.histogram(df, x=selected_feature, nbins=50,
                          title=f"{selected_feature.title()} Distribution",
                          color_discrete_sequence=[colors[0]])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot
        fig = px.box(df, y=selected_feature,
                    title=f"{selected_feature.title()} Box Plot",
                    color_discrete_sequence=[colors[1]])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Distribution by categories
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        selected_category = st.selectbox("Compare distribution by:", categorical_cols)

        fig = px.histogram(df, x=selected_feature, color=selected_category,
                          title=f"{selected_feature.title()} Distribution by {selected_category.title()}",
                          barmode='overlay', opacity=0.7,
                          color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

    # Statistical tests
    st.subheader("📊 Statistical Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean", f"{df[selected_feature].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[selected_feature].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[selected_feature].std():.2f}")

    # Normality test
    stat, p_value = stats.shapiro(df[selected_feature].sample(min(5000, len(df))))
    normality = "Normal" if p_value > 0.05 else "Not Normal"

    st.write(f"**Normality Test:** {normality} (p-value: {p_value:.4f})")

elif page == "🎯 Outliers":
    st.header("🎯 Outlier Analysis")

    # Select feature for outlier analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_feature = st.selectbox("Select Feature for Outlier Analysis:", numeric_cols)

    # IQR method
    Q1 = df[selected_feature].quantile(0.25)
    Q3 = df[selected_feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound)]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Outliers", len(outliers))
    with col2:
        st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.1f}%")
    with col3:
        st.metric("IQR Range", f"{lower_bound:.1f} - {upper_bound:.1f}")

    st.markdown("---")

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(df, y=selected_feature,
                    title=f"{selected_feature.title()} with Outliers",
                    color_discrete_sequence=[colors[0]])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter plot highlighting outliers
        df_copy = df.copy()
        df_copy['is_outlier'] = ((df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound))

        fig = px.scatter(df_copy, x=df.index, y=selected_feature, color='is_outlier',
                        title=f"{selected_feature.title()} Outliers Highlighted",
                        color_discrete_map={False: colors[1], True: 'red'})
        fig.update_layout(xaxis_title="Index", yaxis_title=selected_feature.title())
        st.plotly_chart(fig, use_container_width=True)

    # Outlier details
    if len(outliers) > 0:
        st.subheader("🔍 Outlier Details")
        st.dataframe(outliers[[selected_feature]].describe().round(2), use_container_width=True)

        # Show sample outliers
        st.write("**Sample Outliers:**")
        st.dataframe(outliers.head(10)[[selected_feature]], use_container_width=True)

elif page == "💡 Insights":
    st.header("💡 Key Insights & Recommendations")

    # Calculate key insights
    insights = []

    # Price insights
    avg_price = df['price'].mean()
    median_price = df['price'].median()
    price_skew = df['price'].skew()

    insights.append({
        'category': '💰 Price Analysis',
        'insight': f"Average BMW price is £{avg_price:,.0f} with median £{median_price:,.0f}. Price distribution is {'right-skewed' if price_skew > 0.5 else 'left-skewed' if price_skew < -0.5 else 'approximately normal'} (skewness: {price_skew:.2f})",
        'recommendation': "Consider price segmentation for different market tiers."
    })

    # Model insights
    top_model = df['model'].value_counts().index[0]
    top_model_count = df['model'].value_counts().iloc[0]

    insights.append({
        'category': '🚙 Model Popularity',
        'insight': f"'{top_model}' is the most popular model with {top_model_count} cars ({top_model_count/len(df)*100:.1f}% of dataset).",
        'recommendation': "Focus inventory and marketing efforts on popular models."
    })

    # Fuel type insights
    fuel_efficiency = df.groupby('fuelType')['mpg'].mean().sort_values(ascending=False)
    best_fuel = fuel_efficiency.index[0]

    insights.append({
        'category': '⛽ Fuel Efficiency',
        'insight': f"{best_fuel} cars have the highest average MPG ({fuel_efficiency.iloc[0]:.1f}), indicating better fuel efficiency.",
        'recommendation': "Promote fuel-efficient models for cost-conscious buyers."
    })

    # Year insights
    year_trend = df.groupby('year')['price'].mean()
    price_increase = ((year_trend.iloc[-1] - year_trend.iloc[0]) / year_trend.iloc[0] * 100)

    insights.append({
        'category': '📅 Market Trends',
        'insight': f"Car prices have {'increased' if price_increase > 0 else 'decreased'} by {abs(price_increase):.1f}% from {year_trend.index[0]} to {year_trend.index[-1]}.",
        'recommendation': "Monitor depreciation patterns for pricing strategy."
    })

    # Correlation insights
    price_corr = df.select_dtypes(include=[np.number]).corr()['price'].abs().sort_values(ascending=False)
    top_correlated = price_corr.index[1]  # Skip price itself

    insights.append({
        'category': '🔗 Price Drivers',
        'insight': f"'{top_correlated}' shows the strongest correlation with price among numeric features.",
        'recommendation': "Use this feature prominently in pricing models and customer discussions."
    })

    # Outlier insights
    outlier_percentage = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_percentage[col] = outliers / len(df) * 100

    max_outlier_col = max(outlier_percentage, key=outlier_percentage.get)

    insights.append({
        'category': '🎯 Data Quality',
        'insight': f"'{max_outlier_col}' has the highest outlier percentage ({outlier_percentage[max_outlier_col]:.1f}%).",
        'recommendation': "Review data collection process for this feature."
    })

    # Display insights
    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            <h4>{insight['category']}</h4>
            <p><strong>Insight:</strong> {insight['insight']}</p>
            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Actionable recommendations summary
    st.subheader("🎯 Actionable Recommendations")

    recommendations = [
        "🔸 Implement dynamic pricing based on model popularity and market trends",
        "🔸 Focus marketing on fuel-efficient and popular models",
        "🔸 Develop targeted promotions for different price segments",
        "🔸 Monitor depreciation patterns for better inventory management",
        "🔸 Use data-driven insights for customer recommendations",
        "🔸 Regularly audit data quality and outlier detection"
    ]

    for rec in recommendations:
        st.markdown(rec)

# Footer
st.markdown("---")
st.markdown("🚗 **BMW Car Price Prediction** | Built with Streamlit & Plotly")
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Dashboard Sections:**")
st.sidebar.markdown("- Overview: Dataset summary & statistics")
st.sidebar.markdown("- Price Analysis: Price distributions & trends")
st.sidebar.markdown("- Model Insights: Model popularity & pricing")
st.sidebar.markdown("- Fuel & Transmission: Energy efficiency analysis")
st.sidebar.markdown("- Year Trends: Temporal price patterns")
st.sidebar.markdown("- Correlations: Feature relationships")
st.sidebar.markdown("- Distributions: Statistical analysis")
st.sidebar.markdown("- Outliers: Data quality assessment")
st.sidebar.markdown("- Insights: Key findings & recommendations")