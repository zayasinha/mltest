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
        df = pd.read_csv('notebook/data/raw.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'notebook/data/raw.csv' exists.")
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
        info_df = pd.DataFrame({
            'Feature': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(info_df, use_container_width=True)

    with col2:
        st.subheader("🎲 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

    # Basic statistics
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "📈 Price Analysis":
    st.header("📈 Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig = px.histogram(
            df, x='price', nbins=50,
            title="Price Distribution",
            color_discrete_sequence=[colors[0]]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Price box plot
        fig = px.box(
            df, y='price',
            title="Price Box Plot (Outlier Detection)",
            color_discrete_sequence=[colors[1]]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price by categories
    col1, col2 = st.columns(2)

    with col1:
        # Price by fuel type
        fuel_price = df.groupby('fuelType')['price'].agg(['mean', 'median', 'count']).round(0)
        fig = px.bar(
            fuel_price.reset_index(), x='fuelType', y='mean',
            title="Average Price by Fuel Type",
            color_discrete_sequence=[colors[2]],
            labels={'mean': 'Average Price (£)', 'fuelType': 'Fuel Type'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Price by transmission
        trans_price = df.groupby('transmission')['price'].agg(['mean', 'median', 'count']).round(0)
        fig = px.bar(
            trans_price.reset_index(), x='transmission', y='mean',
            title="Average Price by Transmission",
            color_discrete_sequence=[colors[3]],
            labels={'mean': 'Average Price (£)', 'transmission': 'Transmission'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Price ranges
    st.subheader("💰 Price Ranges Analysis")
    price_ranges = pd.cut(df['price'],
                         bins=[0, 10000, 20000, 30000, 50000, 100000, float('inf')],
                         labels=['<£10k', '£10k-20k', '£20k-30k', '£30k-50k', '£50k-100k', '>£100k'])

    price_dist = df.groupby(price_ranges).size().reset_index(name='count')
    price_dist.columns = ['Price Range', 'Number of Cars']

    fig = px.pie(
        price_dist, values='Number of Cars', names='Price Range',
        title="Car Distribution by Price Ranges",
        color_discrete_sequence=colors
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🚙 Model Insights":
    st.header("🚙 BMW Model Insights")

    # Model distribution
    model_counts = df['model'].value_counts().reset_index()
    model_counts.columns = ['Model', 'Count']

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            model_counts.head(10), x='Model', y='Count',
            title="Top 10 BMW Models by Count",
            color_discrete_sequence=[colors[0]]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Average price by model
        model_prices = df.groupby('model')['price'].agg(['mean', 'count']).round(0)
        model_prices = model_prices.sort_values('mean', ascending=False).head(10)

        fig = px.bar(
            model_prices.reset_index(), x='model', y='mean',
            title="Top 10 Models by Average Price",
            color_discrete_sequence=[colors[1]],
            labels={'mean': 'Average Price (£)', 'model': 'Model'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model year analysis
    st.subheader("📅 Model Age Analysis")
    current_year = 2024
    df['car_age'] = current_year - df['year']

    age_model = df.groupby('model')['car_age'].agg(['mean', 'min', 'max']).round(1)
    age_model = age_model.sort_values('mean').head(10)

    fig = px.bar(
        age_model.reset_index(), x='model', y='mean',
        title="Average Car Age by Model (Years)",
        color_discrete_sequence=[colors[2]],
        labels={'mean': 'Average Age (Years)', 'model': 'Model'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model efficiency analysis
    st.subheader("⛽ Model Efficiency Analysis")
    efficiency = df.groupby('model').agg({
        'mpg': 'mean',
        'engineSize': 'mean',
        'price': 'mean',
        'mileage': 'mean'
    }).round(2)

    # MPG vs Price scatter
    fig = px.scatter(
        efficiency.reset_index(), x='mpg', y='price',
        size='engineSize', color='model',
        title="MPG vs Price by Model (bubble size = engine size)",
        labels={'mpg': 'Miles Per Gallon', 'price': 'Average Price (£)'},
        color_discrete_sequence=colors
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "⛽ Fuel & Transmission":
    st.header("⛽ Fuel Type & Transmission Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Fuel type distribution
        fuel_dist = df['fuelType'].value_counts().reset_index()
        fuel_dist.columns = ['Fuel Type', 'Count']

        fig = px.pie(
            fuel_dist, values='Count', names='Fuel Type',
            title="Fuel Type Distribution",
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Transmission distribution
        trans_dist = df['transmission'].value_counts().reset_index()
        trans_dist.columns = ['Transmission', 'Count']

        fig = px.pie(
            trans_dist, values='Count', names='Transmission',
            title="Transmission Distribution",
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Fuel vs Transmission heatmap
    st.subheader("🔥 Fuel Type vs Transmission Matrix")
    fuel_trans = pd.crosstab(df['fuelType'], df['transmission'])

    fig = px.imshow(
        fuel_trans,
        title="Fuel Type vs Transmission Heatmap",
        color_continuous_scale="Blues",
        labels={'x': 'Transmission', 'y': 'Fuel Type', 'color': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performance by fuel type
    col1, col2 = st.columns(2)

    with col1:
        fuel_perf = df.groupby('fuelType').agg({
            'mpg': 'mean',
            'price': 'mean',
            'mileage': 'mean'
        }).round(2)

        fig = px.bar(
            fuel_perf.reset_index(), x='fuelType', y='mpg',
            title="Average MPG by Fuel Type",
            color_discrete_sequence=[colors[0]],
            labels={'mpg': 'Miles Per Gallon', 'fuelType': 'Fuel Type'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            fuel_perf.reset_index(), x='fuelType', y='price',
            title="Average Price by Fuel Type",
            color_discrete_sequence=[colors[1]],
            labels={'price': 'Average Price (£)', 'fuelType': 'Fuel Type'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📅 Year Trends":
    st.header("📅 Year Trends & Time Analysis")

    # Cars by year
    year_counts = df['year'].value_counts().sort_index().reset_index()
    year_counts.columns = ['Year', 'Count']

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            year_counts, x='Year', y='Count',
            title="Number of Cars by Year",
            markers=True,
            color_discrete_sequence=[colors[0]]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Average price by year
        year_prices = df.groupby('year')['price'].agg(['mean', 'median', 'count']).round(0)
        year_prices = year_prices.reset_index()

        fig = px.line(
            year_prices, x='year', y=['mean', 'median'],
            title="Price Trends by Year",
            markers=True,
            labels={'value': 'Price (£)', 'year': 'Year', 'variable': 'Metric'},
            color_discrete_sequence=[colors[1], colors[2]]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Year and fuel type analysis
    st.subheader("📊 Year vs Fuel Type Trends")
    year_fuel = df.groupby(['year', 'fuelType']).size().reset_index(name='count')

    fig = px.line(
        year_fuel, x='year', y='count', color='fuelType',
        title="Fuel Type Popularity Over Years",
        markers=True,
        labels={'count': 'Number of Cars', 'year': 'Year', 'fuelType': 'Fuel Type'},
        color_discrete_sequence=colors
    )
    st.plotly_chart(fig, use_container_width=True)

    # Engine size trends
    st.subheader("🔧 Engine Size Evolution")
    engine_trends = df.groupby('year')['engineSize'].agg(['mean', 'median']).round(2).reset_index()

    fig = px.line(
        engine_trends, x='year', y=['mean', 'median'],
        title="Engine Size Trends Over Years",
        markers=True,
        labels={'value': 'Engine Size (L)', 'year': 'Year', 'variable': 'Metric'},
        color_discrete_sequence=[colors[3], colors[4]]
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔗 Correlations":
    st.header("🔗 Feature Correlations")

    # Correlation matrix for numerical features
    numerical_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
    corr_matrix = df[numerical_cols].corr()

    # Heatmap
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Numerical Features",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels={'color': 'Correlation'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Strong correlations analysis
    st.subheader("🎯 Key Correlations Analysis")

    # Price correlations
    price_corr = corr_matrix['price'].sort_values(ascending=False)
    price_corr_df = price_corr.reset_index()
    price_corr_df.columns = ['Feature', 'Correlation with Price']

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            price_corr_df[price_corr_df['Feature'] != 'price'],
            x='Feature', y='Correlation with Price',
            title="Feature Correlation with Price",
            color='Correlation with Price',
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter plots for top correlations
        top_corr_features = price_corr.index[1:4]  # Top 3 correlations excluding price itself

        for feature in top_corr_features:
            fig = px.scatter(
                df, x=feature, y='price',
                title=f"Price vs {feature.title()}",
                trendline="ols",
                color_discrete_sequence=[colors[top_corr_features.tolist().index(feature)]]
            )
            st.plotly_chart(fig, use_container_width=True)

    # Categorical correlations
    st.subheader("📊 Categorical Feature Relationships")

    # Transmission vs Fuel Type
    trans_fuel = pd.crosstab(df['transmission'], df['fuelType'], normalize='index') * 100
    trans_fuel = trans_fuel.round(1)

    fig = px.imshow(
        trans_fuel,
        title="Transmission vs Fuel Type (%)",
        color_continuous_scale="Blues",
        labels={'color': 'Percentage (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "📦 Distributions":
    st.header("📦 Feature Distributions")

    numerical_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']

    # Create subplots for distributions
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{col.title()} Distribution" for col in numerical_cols],
        vertical_spacing=0.1
    )

    colors_dist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, col in enumerate(numerical_cols):
        row = (i // 3) + 1
        col_pos = (i % 3) + 1

        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker_color=colors_dist[i],
                showlegend=False
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=600,
        title_text="Numerical Feature Distributions",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Box plots for outlier detection
    st.subheader("📊 Box Plots (Outlier Detection)")

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{col.title()} Box Plot" for col in numerical_cols]
    )

    for i, col in enumerate(numerical_cols):
        row = (i // 3) + 1
        col_pos = (i % 3) + 1

        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                marker_color=colors_dist[i],
                showlegend=False
            ),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=600,
        title_text="Box Plots for Outlier Detection",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Outliers":
    st.header("🎯 Outlier Detection & Analysis")

    numerical_cols = ['price', 'mileage', 'tax', 'mpg', 'engineSize']

    # Z-score method for outlier detection
    st.subheader("📈 Outlier Analysis using Z-Score Method")

    outlier_summary = []

    for col in numerical_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[z_scores > 3]  # Z-score > 3 considered outliers

        outlier_summary.append({
            'Feature': col,
            'Total Values': len(df),
            'Outliers': len(outliers),
            'Outlier Percentage': round(len(outliers) / len(df) * 100, 2),
            'Min Outlier Value': outliers[col].min() if len(outliers) > 0 else None,
            'Max Outlier Value': outliers[col].max() if len(outliers) > 0 else None
        })

    outlier_df = pd.DataFrame(outlier_summary)
    st.dataframe(outlier_df, use_container_width=True)

    st.markdown("---")

    # Price outliers analysis
    st.subheader("💰 Price Outliers Deep Dive")

    # Identify price outliers
    price_z_scores = np.abs(stats.zscore(df['price']))
    price_outliers = df[price_z_scores > 3]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Price Outliers", len(price_outliers))
        st.metric("Outlier Percentage", f"{len(price_outliers)/len(df)*100:.1f}%")

        # Most expensive outliers
        st.subheader("🏆 Most Expensive Outliers")
        expensive_outliers = price_outliers.nlargest(5, 'price')[['model', 'year', 'price', 'mileage', 'fuelType']]
        st.dataframe(expensive_outliers, use_container_width=True)

    with col2:
        # Outlier characteristics
        st.subheader("📊 Outlier Characteristics")
        outlier_stats = price_outliers[['price', 'mileage', 'year', 'engineSize']].describe()
        st.dataframe(outlier_stats.round(2), use_container_width=True)

    # Scatter plot of outliers
    fig = px.scatter(
        df, x='mileage', y='price',
        color=price_z_scores > 3,
        color_discrete_map={True: 'red', False: 'blue'},
        title="Price vs Mileage (Red dots are outliers)",
        labels={'color': 'Is Outlier'}
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "💡 Insights":
    st.header("💡 Key Insights & Findings")

    # Calculate key insights
    insights = []

    # Price insights
    avg_price = df['price'].mean()
    median_price = df['price'].median()
    price_skew = df['price'].skew()

    insights.append({
        'category': '💰 Price Analysis',
        'insight': f"Average BMW price is £{avg_price:,.0f}, median is £{median_price:,.0f}. Price distribution is {'right-skewed' if price_skew > 0 else 'left-skewed'} (skewness: {price_skew:.2f})",
        'impact': 'Most cars are priced below the average, with some luxury models pulling up the mean.'
    })

    # Model insights
    top_model = df['model'].value_counts().index[0]
    top_model_count = df['model'].value_counts().iloc[0]

    insights.append({
        'category': '🚙 Model Popularity',
        'insight': f"The most popular model is {top_model} with {top_model_count} cars ({top_model_count/len(df)*100:.1f}% of total).",
        'impact': 'Focus marketing efforts on popular models while maintaining variety.'
    })

    # Fuel type insights
    diesel_pct = (df['fuelType'] == 'Diesel').mean() * 100
    petrol_pct = (df['fuelType'] == 'Petrol').mean() * 100

    insights.append({
        'category': '⛽ Fuel Preferences',
        'insight': f"Diesel cars: {diesel_pct:.1f}%, Petrol cars: {petrol_pct:.1f}%. Diesel dominates the market.",
        'impact': 'Consider fuel efficiency and environmental factors in inventory decisions.'
    })

    # Year insights
    newest_year = df['year'].max()
    oldest_year = df['year'].min()
    avg_age = 2024 - df['year'].mean()

    insights.append({
        'category': '📅 Age Distribution',
        'insight': f"Cars range from {oldest_year} to {newest_year}, with average age of {avg_age:.1f} years.",
        'impact': 'Balanced inventory of new and pre-owned vehicles.'
    })

    # Correlation insights
    price_year_corr = df['price'].corr(df['year'])
    price_mileage_corr = df['price'].corr(df['mileage'])

    insights.append({
        'category': '🔗 Key Correlations',
        'insight': f"Price correlates positively with year ({price_year_corr:.2f}) and negatively with mileage ({price_mileage_corr:.2f}).",
        'impact': 'Newer cars with lower mileage command higher prices - expected market behavior.'
    })

    # Efficiency insights
    avg_mpg = df['mpg'].mean()
    best_fuel = df.groupby('fuelType')['mpg'].mean().idxmax()
    best_mpg = df.groupby('fuelType')['mpg'].mean().max()

    insights.append({
        'category': '⚡ Efficiency',
        'insight': f"Average MPG: {avg_mpg:.1f}. {best_fuel} cars are most efficient at {best_mpg:.1f} MPG.",
        'impact': 'Highlight fuel efficiency in marketing, especially for diesel models.'
    })

    # Display insights
    for insight in insights:
        st.markdown(f"""
        <div class="insight-box">
            <h4>{insight['category']}</h4>
            <p><strong>Insight:</strong> {insight['insight']}</p>
            <p><strong>Business Impact:</strong> {insight['impact']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations
    st.subheader("🎯 Recommendations")

    recommendations = [
        "📈 **Pricing Strategy**: Focus on mid-range priced vehicles (£20k-£40k) which form the bulk of the market",
        "🚗 **Inventory Management**: Stock more popular models like 3 Series and 1 Series",
        "⛽ **Fuel Mix**: Maintain diesel dominance but consider hybrid options for environmental appeal",
        "📅 **Age Balance**: Keep mix of new and 2-3 year old vehicles for different customer segments",
        "🎯 **Marketing**: Highlight fuel efficiency and reliability of diesel models",
        "💡 **Outlier Management**: Flag extremely high-priced or low-priced vehicles for special handling"
    ]

    for rec in recommendations:
        st.markdown(f"• {rec}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Plotly | Data Source: BMW Car Dataset")
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Dashboard Sections:**")
st.sidebar.markdown("- Overview: Dataset summary & statistics")
st.sidebar.markdown("- Price Analysis: Price distributions & trends")
st.sidebar.markdown("- Model Insights: Model popularity & characteristics")
st.sidebar.markdown("- Fuel & Transmission: Powertrain analysis")
st.sidebar.markdown("- Year Trends: Temporal patterns")
st.sidebar.markdown("- Correlations: Feature relationships")
st.sidebar.markdown("- Distributions: Statistical distributions")
st.sidebar.markdown("- Outliers: Anomaly detection")
st.sidebar.markdown("- Insights: Key findings & recommendations")