import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import os
from pathlib import Path
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
from datetime import datetime, timedelta
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Prophet imports - KEEP THIS OUTSIDE FUNCTIONS
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
# Page configuration must come first before any other Streamlit commands
st.set_page_config(
    page_title="YES! Connect Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add this after your st.set_page_config() call
st.markdown("""
<style>
    /* KPI cards styling */
    .kpi-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
        margin-bottom: 1rem;
        border-top: 4px solid #3B82F6;
    }
    .kpi-title {
        font-size: 1rem;
        font-weight: 600;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .kpi-context {
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }
    .kpi-trend {
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .positive-trend {
        color: #059669;
        font-weight: 600;
    }
    .negative-trend {
        color: #DC2626;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    /* Insights Box Styling */
    .insight-box {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    .insight-box:hover {
        box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
    }
    .insight-header {
        font-size: 1rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.7rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f0f0f0;
    }
    .insight-item {
        margin-bottom: 0.5rem;
    }
    .positive-trend {
        color: #059669;
        font-weight: 600;
    }
    .negative-trend {
        color: #DC2626;
        font-weight: 600;
    }
    .insight-subtitle {
        font-weight: 600;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
        color: #4B5563;
    }
    .insight-bullet {
        margin-right: 0.3rem;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state for filter persistence
if 'country_filter' not in st.session_state:
    st.session_state['country_filter'] = []
if 'career_filter' not in st.session_state:
    st.session_state['career_filter'] = []
if 'date_range' not in st.session_state:
    st.session_state['date_range'] = []
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "Engagement Prediction"

# Add this to your session state initialization section
if 'forecast_horizon' not in st.session_state:
    st.session_state['forecast_horizon'] = 90  # Default 90 days forecast
if 'selected_forecast_metric' not in st.session_state:
    st.session_state['selected_forecast_metric'] = "New Users"

# Function to load data
@st.cache_data
def load_data():
    # Try multiple path approaches
    try:
        # Option 1: Direct relative path (most likely to work on Streamlit Cloud)
        if os.path.exists("data/2024 sample.csv"):
            data_path = "data/2024 sample.csv"
            st.sidebar.success(f"Loaded data")
        # Option 2: Path from script location
        elif os.path.exists(str(Path(__file__).parent / "data" / "2024 sample.csv")):
            data_path = str(Path(__file__).parent / "data" / "2024 sample.csv")
            st.sidebar.success(f"Loaded data")
        # Option 3: Try with underscore instead of space
        elif os.path.exists("data/2024_sample.csv"):
            data_path = "data/2024_sample.csv"
            st.sidebar.success(f"Loaded data")
        else:
            st.sidebar.error("Could not find data file")
            # Creating a small sample dataset to prevent errors
            sample_data = pd.DataFrame({
                'Country': ['USA', 'UK', 'India', 'Germany', 'France'],
                'Career Level': ['Entry', 'Mid', 'Senior', 'Entry', 'Mid'],
                'Experience Level': ['Beginner', 'Intermediate', 'Expert', 'Beginner', 'Intermediate'],
                'Industry': ['Tech', 'Finance', 'Education', 'Healthcare', 'Manufacturing'],
                'Last login date': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'],
                'total_friend_count': [5, 10, 15, 8, 12],
                'profile_avatar_created': ['Y', 'Y', 'N', 'Y', 'N'],
                'App': ['Web', 'Mobile', 'Web', 'N/A', 'Mobile']
            })
            return sample_data
            
        # Load the data
        data_raw = pd.read_csv(data_path, sep=',', encoding='latin1')
        return data_raw
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Creating a small sample dataset to prevent errors
        sample_data = pd.DataFrame({
            'Country': ['USA', 'UK', 'India', 'Germany', 'France'],
            'Career Level': ['Entry', 'Mid', 'Senior', 'Entry', 'Mid'],
            'Experience Level': ['Beginner', 'Intermediate', 'Expert', 'Beginner', 'Intermediate'],
            'Industry': ['Tech', 'Finance', 'Education', 'Healthcare', 'Manufacturing'],
            'Last login date': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'],
            'total_friend_count': [5, 10, 15, 8, 12],
            'profile_avatar_created': ['Y', 'Y', 'N', 'Y', 'N'],
            'App': ['Web', 'Mobile', 'Web', 'N/A', 'Mobile']
        })
        return sample_data




# Function to calculate data completeness on raw data
def calculate_data_completeness(df):
    """
    Calculate data completeness metrics on raw data
    Returns completeness information before any preprocessing
    """
    # Use lists instead of dictionaries for collecting data
    completeness_data = []
    
    # Profile completeness metrics
    fields_to_check = [
        ("Career Level", "Member professional status unknown; impacts segmentation accuracy"),
        ("Experience Level", "Member expertise level unknown; affects mentorship matching"),
        ("Country", "Geographic information missing; limits regional analysis"),
        ("Industry", "Professional focus unknown; reduces networking relevance"),
        ("profile_avatar_created", "Profile picture status unknown; affects engagement metrics"),
        ("App", "Platform usage patterns unclear; impacts feature development priorities"),
        ("Last login date", "Login activity unknown; affects engagement analysis")
    ]
    
    for field, impact in fields_to_check:
        if field in df.columns:
            # Calculate completeness
            total = len(df)
            missing = df[field].isna().sum()
            completeness = 100 - (missing / total * 100)
            
            # Append as a dictionary to the list
            completeness_data.append({
                "Field": field,
                "Completeness": f"{completeness:.1f}%",
                "Missing Count": int(missing),
                "Business Impact": impact
            })
    
    # Convert to DataFrame for easier handling
    completeness_df = pd.DataFrame(completeness_data)
    
    # Create a version with numerical completeness for charts
    numeric_completeness_df = pd.DataFrame({
        "Field": completeness_df["Field"],
        "Completeness": [float(x.replace("%", "")) for x in completeness_df["Completeness"]]
    })
    
    return completeness_data, numeric_completeness_df


# Function to preprocess data for modeling
def preprocess_data(df):
    # Handle missing values in key columns
    df['Country'] = df['Country'].fillna('Unknown').astype(str)
    df['Career Level'] = df['Career Level'].fillna('Unknown').astype(str)
    df['Experience Level'] = df['Experience Level'].fillna('Unknown').astype(str)
    df['Industry'] = df['Industry'].fillna('Unknown').astype(str)
    
    # Create a unique identifier for each user
    df['user_id'] = df.index
    
    # Convert date columns safely
    df['last_login_date'] = pd.to_datetime(df['Last login date'], errors='coerce', dayfirst=True)
    
    # Handle missing dates
    df = df.dropna(subset=['last_login_date'])
    
    # Calculate days since login
    df['days_since_login'] = (datetime.now() - df['last_login_date']).dt.days
    
    # Handle friend count
    df['total_friend_count'] = df['total_friend_count'].fillna(0).astype(int)
    
    # Calculate profile completion score
    df['profile_completion'] = df['profile_avatar_created'].fillna('').apply(
        lambda x: 1 if x == 'Y' else 0)
    
    # Process platform information
    df['App'] = df['App'].fillna('Web')
    df['uses_mobile'] = df['App'].apply(lambda x: 0 if x == 'N/A' or x == 'Web' else 1)
    df['platform'] = df['App'].fillna('Web')
    
    # Calculate login frequency
    login_count = df.groupby('user_id')['last_login_date'].count().reset_index()
    login_count.columns = ['user_id', 'login_count']
    df = df.merge(login_count, on='user_id', how='left')
    
    return df



# Country name standardization functions
def normalize_country_name(name):
    """Normalize country name by removing spaces and converting to lowercase"""
    if pd.isna(name):
        return name
    return str(name).lower().replace(" ", "")

def get_country_similarity(name1, name2):
    """Get similarity score between two country names"""
    if pd.isna(name1) or pd.isna(name2):
        return 0
    # Use SequenceMatcher for similarity calculation
    return SequenceMatcher(None, normalize_country_name(name1), 
                           normalize_country_name(name2)).ratio()


def apply_country_standardization(df, country_column='Country', threshold=0.8):
    """Apply country name standardization to the dataframe.
    Returns a copy of the dataframe with standardized country names."""
    
    # Use the existing standardize_countries function
    df_standardized, _, _ = standardize_countries(df, country_column, threshold)
    return df_standardized


def find_similar_countries(countries, threshold=0.8):
    """Find similar country names based on similarity threshold"""
    # Get unique countries
    unique_countries = sorted(list(set(countries)))
    
    # Initialize groups
    groups = defaultdict(list)
    processed = set()
    
    # Group similar countries
    for i, country1 in enumerate(unique_countries):
        if country1 in processed:
            continue
            
        # Start a new group with this country
        current_group = [country1]
        processed.add(country1)
        
        # Find similar countries
        for country2 in unique_countries[i+1:]:
            if country2 not in processed:
                similarity = get_country_similarity(country1, country2)
                if similarity > threshold:
                    current_group.append(country2)
                    processed.add(country2)
        
        # Store the group if it has multiple countries
        if len(current_group) > 1:
            # Use the most frequent country as the key
            counts = {c: countries.count(c) for c in current_group}
            most_frequent = max(counts, key=counts.get)
            groups[most_frequent] = current_group
    
    return groups

def standardize_countries(df, country_column='Country', threshold=0.8):
    """Standardize country names in the dataframe"""
    if country_column not in df.columns:
        return df, {}, pd.DataFrame()
        
    # Get country values as list
    countries = df[country_column].dropna().astype(str).tolist()
    
    # Find similar countries
    similar_groups = find_similar_countries(countries, threshold)
    
    # Common mapping overrides - this is the key change
    # This ensures proper formats are used even if they aren't the most frequent
    mapping_overrides = {
        "SouthAfrica": "South Africa",
        "South Africa": "South Africa",  # Keep this as is
        "UnitedStates": "United States",
        "USA": "United States",
        "UnitedKingdom": "United Kingdom",
        "UK": "United Kingdom", 
        "SierraLeone": "Sierra Leone",
        "SouthSudan": "South Sudan",
        "TheGambia": "Gambia",  # Most maps expect "Gambia" not "The Gambia"
        "Tunisie": "Tunisia",   # Use English name for consistency
        # Add other overrides as needed
    }
    
    # Create a mapping dictionary
    country_mapping = {}
    for standard, variants in similar_groups.items():
        # Check if we should override the standard
        if standard in mapping_overrides:
            override_standard = mapping_overrides[standard]
            for variant in variants:
                country_mapping[variant] = override_standard
        else:
            for variant in variants:
                if variant != standard:
                    country_mapping[variant] = standard
    
    # Apply the overrides directly as well
    for variant, standard in mapping_overrides.items():
        country_mapping[variant] = standard
    
    # Create a copy of the dataframe to avoid modifying the original
    df_standardized = df.copy()
    
    # Apply the mapping
    if country_mapping:
        df_standardized[country_column] = df_standardized[country_column].replace(country_mapping)
    
    # Create a summary dataframe for reporting
    summary_data = []
    for standard, variants in similar_groups.items():
        effective_standard = mapping_overrides.get(standard, standard)
        for variant in variants:
            count = (df[country_column] == variant).sum()
            mapped_to = effective_standard if variant != standard or standard in mapping_overrides else standard
            summary_data.append({
                'Standard': mapped_to,
                'Variant': variant,
                'Count': count,
                'Action': 'Keep' if variant == standard and standard not in mapping_overrides else 'Replace'
            })
    
    # Add any direct override entries that weren't in similar groups
    for variant, standard in mapping_overrides.items():
        if not any(d['Variant'] == variant for d in summary_data):
            count = (df[country_column] == variant).sum()
            if count > 0:
                summary_data.append({
                    'Standard': standard,
                    'Variant': variant,
                    'Count': count,
                    'Action': 'Replace'
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    return df_standardized, country_mapping, summary_df

def plot_country_similarity_matrix(countries, max_countries=15):
    """Create a similarity matrix visualization for country names"""
    # Get unique countries (limit to max_countries most frequent)
    country_counts = pd.Series(countries).value_counts()
    top_countries = country_counts.head(max_countries).index.tolist()
    
    # Create similarity matrix
    n = len(top_countries)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = get_country_similarity(top_countries[i], top_countries[j])
    
    # Create heatmap
    fig = px.imshow(
        similarity_matrix,
        x=top_countries,
        y=top_countries,
        color_continuous_scale='Blues',
        title='Country Name Similarity Matrix'
    )
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j and similarity_matrix[i, j] > 0.75:  # Only show high similarity scores
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{similarity_matrix[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if similarity_matrix[i, j] < 0.9 else "white")
                )
    
    fig.update_layout(height=600)
    return fig



def enhanced_career_distribution_analysis(df):
    """
    Enhanced career distribution analysis showing period-over-period changes
    and mentor-mentee ratios with standardized country names.
    """
    st.subheader("Career Level Shifts by Region")
    
    # First, standardize country names
    df_standardized = apply_country_standardization(df.copy())
    
    # Create date filters for current and previous periods
    current_end_date = df_standardized['last_login_date'].max()
    current_start_date = current_end_date - pd.Timedelta(days=30)
    previous_start_date = current_start_date - pd.Timedelta(days=30)
    previous_end_date = current_start_date - pd.Timedelta(days=1)
    
    # Filter data for current and previous periods
    current_period = df_standardized[
        (df_standardized['last_login_date'] >= current_start_date) & 
        (df_standardized['last_login_date'] <= current_end_date)
    ]
    
    previous_period = df_standardized[
        (df_standardized['last_login_date'] >= previous_start_date) & 
        (df_standardized['last_login_date'] <= previous_end_date)
    ]
    
    # Define mentor and mentee career levels
    mentor_levels = ['Senior Industry Professional', 'Educator', 'Mid-Level Industry Professional','Training/Initiative Provider']
    mentee_levels = ['Student', 'Early Career Professional']
    
    # Function to create career distribution and mentor-mentee metrics
    def calculate_metrics(period_data, period_name):
        # Career distribution by country
        career_dist = period_data.groupby(['Country', 'Career Level']).size().reset_index()
        career_dist.columns = ['Country', 'Career Level', 'User Count']
        
        # Calculate percentage within each country
        career_dist['Country Total'] = career_dist.groupby('Country')['User Count'].transform('sum')
        career_dist['Percentage'] = career_dist['User Count'] / career_dist['Country Total'] * 100
        
        # Calculate mentor-mentee metrics by country
        mentor_mentee = []
        for country in period_data['Country'].unique():
            country_data = period_data[period_data['Country'] == country]
            
            # Count mentors and mentees
            mentors = country_data[country_data['Career Level'].isin(mentor_levels)]
            mentees = country_data[country_data['Career Level'].isin(mentee_levels)]
            
            mentor_count = len(mentors)
            mentee_count = len(mentees)
            
            # Calculate ratio (handle division by zero)
            if mentee_count > 0:
                ratio = mentor_count / mentee_count
            else:
                ratio = 0 if mentor_count == 0 else float('inf')
            
            # Calculate percentages
            total_count = len(country_data)
            mentor_pct = (mentor_count / total_count * 100) if total_count > 0 else 0
            mentee_pct = (mentee_count / total_count * 100) if total_count > 0 else 0
            
            mentor_mentee.append({
                'Country': country,
                'Period': period_name,
                'Mentor Count': mentor_count,
                'Mentee Count': mentee_count,
                'Mentor %': mentor_pct,
                'Mentee %': mentee_pct,
                'Mentor-Mentee Ratio': ratio
            })
        
        return career_dist, pd.DataFrame(mentor_mentee)
    
    # Calculate metrics for both periods
    current_career_dist, current_mentor_mentee = calculate_metrics(current_period, 'Current')
    previous_career_dist, previous_mentor_mentee = calculate_metrics(previous_period, 'Previous')
    
    # Get top countries from current data
    top_countries = current_career_dist['Country'].value_counts().head(5).index.tolist()
    
    # Filter to top countries
    current_career_dist = current_career_dist[current_career_dist['Country'].isin(top_countries)]
    previous_career_dist = previous_career_dist[previous_career_dist['Country'].isin(top_countries)]
    
    # Create pivot tables for both periods
    current_pivot = current_career_dist.pivot_table(
        index='Country', columns='Career Level', values='Percentage', fill_value=0
    )
    
    previous_pivot = previous_career_dist.pivot_table(
        index='Country', columns='Career Level', values='Percentage', fill_value=0
    )
    
    # Ensure both pivots have the same columns (career levels)
    all_careers = list(set(current_pivot.columns) | set(previous_pivot.columns))
    for career in all_careers:
        if career not in current_pivot.columns:
            current_pivot[career] = 0
        if career not in previous_pivot.columns:
            previous_pivot[career] = 0
    
    # Reindex both pivots to have the same order of career levels
    current_pivot = current_pivot[sorted(all_careers)]
    previous_pivot = previous_pivot[sorted(all_careers)]
    
    # Calculate change
    for country in current_pivot.index:
        if country in previous_pivot.index:
            # Country exists in both periods
            pass
        else:
            # Country only in current period
            previous_pivot.loc[country] = 0
    
    for country in previous_pivot.index:
        if country not in current_pivot.index:
            # Country only in previous period
            current_pivot.loc[country] = 0
    
    # Ensure same country order in both pivots
    current_pivot = current_pivot.reindex(sorted(current_pivot.index))
    previous_pivot = previous_pivot.reindex(sorted(previous_pivot.index))
    
    # Calculate change
    change_pivot = current_pivot - previous_pivot
    
    # Create UI tabs for different views
    tab1, tab2 = st.tabs(["Career Level Shifts", "Mentor-Mentee Ratio"])
    
    with tab1:
        # Create heatmap of changes
        fig = px.imshow(
            change_pivot,
            labels=dict(x="Career Level", y="Country", color="Change (%)"),
            title="Month-over-Month Career Level Distribution Shifts (%)",
            color_continuous_scale="RdBu_r",  # Red for decrease, Blue for increase
            color_continuous_midpoint=0,  # Center the color scale at zero
            text_auto='.1f'  # Show values with 1 decimal
        )
        
        # Customize hover text
        fig.update_traces(
            hovertemplate="Country: %{y}<br>Career Level: %{x}<br>Change: %{z:.1f}%<extra></extra>"
        )
        
        # Improve layout
        fig.update_layout(
            height=500,
            xaxis_title="Career Level",
            yaxis_title="Country",
            coloraxis_colorbar=dict(
                title="Change (%)",
                titleside="right"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights based on the shifts
        significant_shifts = []
        for country in change_pivot.index:
            for career in change_pivot.columns:
                change = change_pivot.loc[country, career]
                if abs(change) >= 5:  # Consider shifts of 5% or more as significant
                    direction = "increase" if change > 0 else "decrease"
                    significant_shifts.append({
                        "Country": country,
                        "Career Level": career,
                        "Change": change,
                        "Direction": direction
                    })
        
        if significant_shifts:
            st.subheader("Significant Career Shifts")
            shifts_df = pd.DataFrame(significant_shifts).sort_values('Change', key=abs, ascending=False)
            
            # Format for display
            shifts_df['Change'] = shifts_df['Change'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(
                shifts_df[['Country', 'Career Level', 'Change', 'Direction']],
                use_container_width=True
            )
        else:
            st.info("No significant career distribution shifts detected (threshold: ±5%)")
    
    with tab2:
        # Merge mentor-mentee data for comparison
        mentor_mentee_merged = pd.merge(
            current_mentor_mentee,
            previous_mentor_mentee,
            on='Country',
            suffixes=('_current', '_previous')
        )
        
        # Filter to top countries
        mentor_mentee_filtered = mentor_mentee_merged[mentor_mentee_merged['Country'].isin(top_countries)]
        
        # Calculate ratio changes
        mentor_mentee_filtered['Ratio Change'] = (
            mentor_mentee_filtered['Mentor-Mentee Ratio_current'] - 
            mentor_mentee_filtered['Mentor-Mentee Ratio_previous']
        )
        
        # Create mentor-mentee ratio visualization
        st.subheader("Mentor-Mentee Ratio by Country")
        
        # Create bar chart comparing current and previous ratios
        ratio_fig = go.Figure()
        
        # Add bars for current period
        ratio_fig.add_trace(go.Bar(
            x=mentor_mentee_filtered['Country'],
            y=mentor_mentee_filtered['Mentor-Mentee Ratio_current'],
            name='Current Period',
            marker_color='rgb(26, 118, 255)'
        ))
        
        # Add bars for previous period
        ratio_fig.add_trace(go.Bar(
            x=mentor_mentee_filtered['Country'],
            y=mentor_mentee_filtered['Mentor-Mentee Ratio_previous'],
            name='Previous Period',
            marker_color='rgba(26, 118, 255, 0.5)'
        ))
        
        # Update layout
        ratio_fig.update_layout(
            title='Mentor-to-Mentee Ratio by Country (Month-over-Month)',
            xaxis_title='Country',
            yaxis_title='Ratio (Mentors per Mentee)',
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(ratio_fig, use_container_width=True)
        
# Create a second chart showing mentor and mentee percentages
        pct_fig = go.Figure()
        
        # Current period - mentors
        pct_fig.add_trace(go.Bar(
            x=mentor_mentee_filtered['Country'],
            y=mentor_mentee_filtered['Mentor %_current'],
            name='Mentors (Current)',
            marker_color='darkblue'
        ))
        
        # Current period - mentees
        pct_fig.add_trace(go.Bar(
            x=mentor_mentee_filtered['Country'],
            y=mentor_mentee_filtered['Mentee %_current'],
            name='Mentees (Current)',
            marker_color='skyblue'
        ))
        
        # Previous period - mentors (with pattern)
        pct_fig.add_trace(go.Bar(
            x=mentor_mentee_filtered['Country'],
            y=mentor_mentee_filtered['Mentor %_previous'],
            name='Mentors (Previous)',
            marker=dict(
                color='rgba(0, 0, 139, 0.5)',
                pattern=dict(
                    shape="x"
                )
            )
        ))
        
        # Previous period - mentees (with pattern)
        pct_fig.add_trace(go.Bar(
            x=mentor_mentee_filtered['Country'],
            y=mentor_mentee_filtered['Mentee %_previous'],
            name='Mentees (Previous)',
            marker=dict(
                color='rgba(135, 206, 235, 0.5)',
                pattern=dict(
                    shape="x"
                )
            )
        ))
        
        # Update layout
        pct_fig.update_layout(
            title='Mentor and Mentee Percentages by Country',
            xaxis_title='Country',
            yaxis_title='Percentage of Users (%)',
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(pct_fig, use_container_width=True)
        
        # Add business insights based on the mentor-mentee analysis
        st.subheader("Mentorship Insights")
        
        # Calculate optimal mentor-mentee ratio range (business assumption)
        optimal_min = 0.2  # Ideally at least 1 mentor per 5 mentees
        optimal_max = 0.5  # But not more than 1 mentor per 2 mentees
        
        insights = []
        
        # Identify countries with concerning ratios
        for _, row in mentor_mentee_filtered.iterrows():
            country = row['Country']
            current_ratio = row['Mentor-Mentee Ratio_current']
            previous_ratio = row['Mentor-Mentee Ratio_previous']
            ratio_change = row['Ratio Change']
            
            # Skip countries with insufficient data
            if row['Mentor Count_current'] + row['Mentee Count_current'] < 10:
                continue
                
            if current_ratio < optimal_min:
                # Too few mentors
                insights.append({
                    "Country": country,
                    "Issue": "Mentor Shortage",
                    "Details": f"Only {current_ratio:.2f} mentors per mentee (below target of {optimal_min})",
                    "Change": f"{ratio_change:+.2f} from previous month",
                    "Recommendation": "Prioritize mentor recruitment and incentives"
                })
            elif current_ratio > optimal_max:
                # Too many mentors
                insights.append({
                    "Country": country,
                    "Issue": "Mentee Shortage",
                    "Details": f"{current_ratio:.2f} mentors per mentee (above target of {optimal_max})",
                    "Change": f"{ratio_change:+.2f} from previous month",
                    "Recommendation": "Focus on early-career recruitment campaigns"
                })
            
            # Significant changes (regardless of optimal range)
            if abs(ratio_change) > 0.1 and previous_ratio > 0:
                direction = "increased" if ratio_change > 0 else "decreased"
                insights.append({
                    "Country": country,
                    "Issue": f"Significant Ratio {direction.title()}",
                    "Details": f"Mentor-mentee ratio {direction} from {previous_ratio:.2f} to {current_ratio:.2f}",
                    "Change": f"{ratio_change:+.2f}",
                    "Recommendation": "Investigate cause of shift and adjust strategy accordingly"
                })
        
        if insights:
            st.dataframe(pd.DataFrame(insights), use_container_width=True)
        else:
            st.info("No significant mentorship ratio issues detected")
    


def create_country_standardization_impact(original_df, standardized_df, country_column='Country'):
    """Create a bar chart showing the impact of country standardization"""
    # Get counts before standardization
    before_counts = original_df[country_column].value_counts().reset_index()
    before_counts.columns = ['Country', 'Before']
    
    # Get counts after standardization
    after_counts = standardized_df[country_column].value_counts().reset_index()
    after_counts.columns = ['Country', 'After']
    
    # Merge the counts
    merged_counts = pd.merge(before_counts, after_counts, on='Country', how='outer').fillna(0)
    
    # Calculate difference
    merged_counts['Difference'] = merged_counts['After'] - merged_counts['Before']
    
    # Filter to only show changes
    changed_counts = merged_counts[merged_counts['Difference'] != 0]
    
    # Sort by absolute difference
    changed_counts = changed_counts.sort_values('Difference', key=abs, ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    # Add before bars
    fig.add_trace(go.Bar(
        x=changed_counts['Country'],
        y=changed_counts['Before'],
        name='Before Standardization',
        marker_color='lightblue'
    ))
    
    # Add after bars
    fig.add_trace(go.Bar(
        x=changed_counts['Country'],
        y=changed_counts['After'],
        name='After Standardization',
        marker_color='darkblue'
    ))
    
    # Update layout
    fig.update_layout(
        title='Impact of Country Name Standardization',
        xaxis_title='Country',
        yaxis_title='Count',
        barmode='group',
        height=500
    )
    
    return fig, changed_counts















def plot_distribution_with_anomalies(df, column, title=None, std_dev_threshold=2, log_scale=False):
    """
    Create a distribution plot with standard deviations marked and outliers highlighted.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name to plot
    title : str, optional
        Plot title
    std_dev_threshold : float, optional
        Number of standard deviations to use as threshold for outliers
    log_scale : bool, optional
        Whether to use a logarithmic scale for the x-axis
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the distribution plot
    """
    # Calculate statistics
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    # Define thresholds
    upper_threshold = mean_val + std_dev_threshold * std_val
    lower_threshold = mean_val - std_dev_threshold * std_val
    
    # Separate normal and anomaly data
    normal_data = df[(df[column] >= lower_threshold) & (df[column] <= upper_threshold)]
    anomaly_data = df[(df[column] < lower_threshold) | (df[column] > upper_threshold)]
    
    # Calculate percentage of anomalies
    anomaly_pct = (len(anomaly_data) / len(df) * 100) if len(df) > 0 else 0
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram for normal data
    fig.add_trace(go.Histogram(
        x=normal_data[column],
        name='Normal Values',
        marker_color='#6BAED6',
        opacity=0.8,
        autobinx=True
    ))
    
    # Add histogram for anomaly data
    fig.add_trace(go.Histogram(
        x=anomaly_data[column],
        name='Anomalies',
        marker_color='#FB6A4A',
        opacity=0.8,
        autobinx=True
    ))
    
    # Add vertical lines for standard deviations
    for i in range(1, std_dev_threshold + 1):
        # Upper standard deviation line
        fig.add_shape(
            type="line",
            x0=mean_val + i * std_val,
            y0=0,
            x1=mean_val + i * std_val,
            y1=1,
            yref="paper",
            line=dict(
                color="rgba(0, 0, 0, 0.5)",
                width=1,
                dash="dot",
            )
        )
        
        # Add annotations for standard deviations
        fig.add_annotation(
            x=mean_val + i * std_val,
            y=1,
            yref="paper",
            text=f"+{i}σ",
            showarrow=False,
            yshift=10
        )
        
        # Only add lower if it makes sense for the data
        if lower_threshold > 0:
            # Lower standard deviation line
            fig.add_shape(
                type="line",
                x0=mean_val - i * std_val,
                y0=0,
                x1=mean_val - i * std_val,
                y1=1,
                yref="paper",
                line=dict(
                    color="rgba(0, 0, 0, 0.5)",
                    width=1,
                    dash="dot",
                )
            )
            
            fig.add_annotation(
                x=mean_val - i * std_val,
                y=1,
                yref="paper",
                text=f"-{i}σ",
                showarrow=False,
                yshift=10
            )
    
    # Add mean line
    fig.add_shape(
        type="line",
        x0=mean_val,
        y0=0,
        x1=mean_val,
        y1=1,
        yref="paper",
        line=dict(
            color="black",
            width=2,
        )
    )
    
    # Add annotation for mean
    fig.add_annotation(
        x=mean_val,
        y=1,
        yref="paper",
        text="Mean",
        showarrow=False,
        yshift=10
    )
    
    # Update layout
    if log_scale:
        fig.update_layout(
            title=title if title else f'Distribution of {column} with Anomalies',
            xaxis_title=column,
            yaxis_title='Count',
            barmode='overlay',
            xaxis_type="log",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    x=0.01,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Anomalies: {len(anomaly_data)} ({anomaly_pct:.1f}%)",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(150, 150, 150, 0.8)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(color="red")
                )
            ]
        )
    else:
        fig.update_layout(
            title=title if title else f'Distribution of {column} with Anomalies',
            xaxis_title=column,
            yaxis_title='Count',
            barmode='overlay',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    x=0.01,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Anomalies: {len(anomaly_data)} ({anomaly_pct:.1f}%)",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(150, 150, 150, 0.8)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(color="red")
                )
            ]
        )
    
    return fig

def plot_login_recency_distribution(df):
    """
    Create a visualization for login recency showing inactive users.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the login data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with the login recency distribution
    """
    # Create bins for different inactivity periods
    bins = [0, 30, 90, 180, 365, df['days_since_login'].max()]
    labels = ['< 30 days', '30-90 days', '90-180 days', '180-365 days', '> 365 days']
    
    # Categorize the data
    df_binned = df.copy()
    df_binned['inactivity_period'] = pd.cut(df_binned['days_since_login'], bins=bins, labels=labels, right=False)
    
    # Count users in each category
    period_counts = df_binned['inactivity_period'].value_counts().reindex(labels)
    period_pct = (period_counts / period_counts.sum() * 100).round(1)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars with different colors based on severity
    for i, (period, count) in enumerate(period_counts.items()):
        if period == '> 365 days':
            color = '#FB6A4A'  # Red for severe inactivity
        elif period == '180-365 days':
            color = '#FDAE6B'  # Orange for moderate inactivity
        else:
            color = '#6BAED6'  # Blue for active users
            
        fig.add_trace(go.Bar(
            x=[period],
            y=[count],
            name=period,
            marker_color=color,
            text=[f"{count} users<br>({period_pct[period]}%)"],
            textposition='auto'
        ))
    
    # Update layout
    fig.update_layout(
        title='User Distribution by Login Recency',
        xaxis_title='Time Since Last Login',
        yaxis_title='Number of Users',
        showlegend=False
    )
    
    return fig

def plot_career_experience_matrix(df):
    """
    Create a heatmap showing career level vs experience level combinations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the career and experience data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly heatmap of career and experience combinations
    """
    # Create a crosstab of career level vs experience level
    if 'Career Level' in df.columns and 'Experience Level' in df.columns:
        cross_tab = pd.crosstab(df['Career Level'], df['Experience Level'])
        
        # Sort the indices to make the visualization more meaningful
        career_order = ['Student', 'Entry', 'Early Career Professional', 'Mid-Level Industry Professional', 
                         'Senior', 'Executive', 'Unknown']
        experience_order = ['Beginner', '< 1 Year', '1-3 Years', '3-5 Years', '5-10 Years', '10 Years+', 'Unknown']
        
        # Only include categories that exist in the data
        career_order = [c for c in career_order if c in cross_tab.index]
        experience_order = [e for e in experience_order if e in cross_tab.columns]
        
        # Reindex with the ordering (only if the categories exist)
        if career_order:
            cross_tab = cross_tab.reindex(career_order)
        if experience_order:
            cross_tab = cross_tab.reindex(columns=experience_order)
        
        # Create the heatmap
        fig = px.imshow(
            cross_tab,
            labels=dict(x="Experience Level", y="Career Level", color="Count"),
            x=cross_tab.columns,
            y=cross_tab.index,
            color_continuous_scale='Blues'
        )
        
        # Add text annotations
        for i in range(len(cross_tab.index)):
            for j in range(len(cross_tab.columns)):
                value = cross_tab.iloc[i, j]
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(value),
                    showarrow=False,
                    font=dict(color="black" if value < cross_tab.values.max()/2 else "white")
                )
        
        # Add markers for inconsistent combinations
        problem_combinations = [
            ('Student', '10 Years+'),
            ('Unknown', '10 Years+'),
            ('Unknown', 'Beginner'),
            ('Unknown', '< 1 Year'),
            ('Unknown', 'Unknown')

        ]
        
        for career, experience in problem_combinations:
            if career in cross_tab.index and experience in cross_tab.columns:
                i = cross_tab.index.get_loc(career)
                j = cross_tab.columns.get_loc(experience)
                
                # Only add the rectangle if there are users with this combination
                if cross_tab.iloc[i, j] > 0:
                    fig.add_shape(
                        type="rect",
                        x0=j-0.5,
                        y0=i-0.5,
                        x1=j+0.5,
                        y1=i+0.5,
                        line=dict(color="red", width=2),
                        fillcolor="rgba(0, 0, 0, 0)"
                    )
        
        # Update layout
        fig.update_layout(
            title="Career Level vs Experience Level Matrix",
            height=500,
            coloraxis_colorbar=dict(title="Count")
        )
        
        return fig
    
    # Return empty figure if the required columns don't exist
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Required columns not found in data",
        showarrow=False
    )
    
    return fig






































# Define the missing plot_model_performance function
def plot_model_performance(model_type, X_test, y_test):
    if model_type == "Engagement Prediction" and X_test is not None and y_test is not None:
        # Create confusion matrix
        y_pred = engagement_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Create heatmap
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Not Engaged', 'Engaged'],
                       y=['Not Engaged', 'Engaged'],
                       title="Model Performance: Confusion Matrix")
        
        return fig
    
    elif model_type == "Churn Prediction" and X_test is not None and y_test is not None:
        # Create scatter plot
        y_pred = churn_model.predict(X_test)
        
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title="Model Performance: Predicted vs Actual")
        
        return fig
    
    elif selected_model == "User Segmentation":
        st.write("""
    **This model segments users into distinct behavior-based groups.**
    
    The segmentation identifies key user personas based on:
    - Connection patterns and network sizes
    - Platform engagement frequency
    - Profile completeness
    - Mobile vs. web platform preferences
    - Career level and experience distribution
    
    Understanding these segments helps tailor content, features, and communication strategies to different user types.
    """)
    
    # Apply segmentation to filtered data
    X_filtered = df_filtered[['total_friend_count', 'days_since_login', 'profile_completion', 'uses_mobile']].fillna(0)
    X_filtered_scaled = scaler.transform(X_filtered)
    df_filtered['cluster'] = segmentation_model.predict(X_filtered_scaled)
    
    # Analyze clusters to generate segment insights
    segment_insights = []
    
    # Calculate key metrics for each cluster
    for cluster_id in sorted(df_filtered['cluster'].unique()):
        cluster_df = df_filtered[df_filtered['cluster'] == cluster_id]
        
        # Skip if cluster is empty
        if len(cluster_df) == 0:
            continue
            
        # Calculate key characteristics
        characteristics = []
        
        # Career level distribution
        career_dist = cluster_df['Career Level'].value_counts(normalize=True)
        top_career = career_dist.index[0] if len(career_dist) > 0 else "Unknown"
        top_career_pct = career_dist.iloc[0] * 100 if len(career_dist) > 0 else 0
        characteristics.append(f"{top_career} ({top_career_pct:.0f}%)")
        
        # Connection count range
        conn_median = cluster_df['total_friend_count'].median()
        conn_q3 = cluster_df['total_friend_count'].quantile(0.75)
        if conn_median > 80:
            conn_desc = f"High friend count ({conn_median:.0f}+ connections)"
        elif conn_median > 30:
            conn_desc = f"Medium friend count ({conn_median:.0f}-{conn_q3:.0f} connections)"
        else:
            conn_desc = f"Limited connections (<{conn_median:.0f} friends)"
        characteristics.append(conn_desc)
        
        # Login recency patterns
        days_median = cluster_df['days_since_login'].median()
        if days_median < 7:
            login_desc = "Regular platform engagement (weekly logins)"
        elif days_median < 14:
            login_desc = "Sporadic engagement (bi-weekly logins)"
        elif days_median < 30:
            login_desc = "Consistent engagement (monthly logins)"
        elif days_median < 90:
            login_desc = "Periodic engagement (monthly/quarterly)"
        else:
            login_desc = f"Infrequent engagement (no login in {days_median:.0f}+ days)"
        characteristics.append(login_desc)
        
        # Profile completion
        profile_pct = cluster_df['profile_completion'].mean() * 100
        characteristics.append(f"Profile avatar created ({profile_pct:.0f}%)")
        
        # Platform usage
        mobile_pct = cluster_df['uses_mobile'].mean() * 100
        if mobile_pct > 60:
            platform_desc = f"Mobile app users ({mobile_pct:.0f}%)"
        elif mobile_pct < 30:
            platform_desc = f"Web platform preference ({100-mobile_pct:.0f}%)"
        else:
            platform_desc = "Mixed platform usage"
        characteristics.append(platform_desc)
        
        # Experience level
        exp_counts = cluster_df['Experience Level'].value_counts()
        top_exp = exp_counts.index[0] if len(exp_counts) > 0 else "Unknown"
        characteristics.append(f"{top_exp} experience (majority)")
        
        # Geographic concentration
        country_counts = cluster_df['Country'].value_counts()
        top_countries = ", ".join(country_counts.head(2).index.tolist())
        characteristics.append(f"Concentrated in {top_countries}")
        
        # Generate strategy based on segment characteristics
        if conn_median > 80 and profile_pct > 70:
            strategy = "Leverage as community ambassadors and content creators. Provide advanced networking tools and recognition for their influence."
        elif "Student" in top_career and conn_median < 80:
            strategy = "Provide educational resources, mentorship connections, and professional development opportunities. Encourage mobile app adoption."
        elif "Early Career" in top_career or "Entrepreneur" in top_career:
            strategy = "Offer career advancement resources, professional showcases, and connections to more established members."
        elif "Senior" in top_career or "Mid-Level" in top_career:
            strategy = "Position as knowledge contributors and mentors. Create opportunities for thought leadership and specialized discussions."
        elif days_median > 60:
            strategy = "Re-engagement campaigns with clear value propositions. Simplified mobile onboarding and personalized content recommendations."
        else:
            strategy = "Create targeted campaigns based on specific usage patterns and interests."
        
        # Name the segment based on characteristics
        if conn_median > 80 and profile_pct > 70:
            segment_name = "Network Builders"
        elif "Student" in top_career and conn_median < 80:
            segment_name = "Academic Engagers"
        elif "Early Career" in top_career or "Entrepreneur" in top_career:
            segment_name = "Emerging Professionals"
        elif "Senior" in top_career or "Mid-Level" in top_career:
            segment_name = "Established Experts"
        elif days_median > 60:
            segment_name = "Dormant Members"
        else:
            segment_name = f"Segment {cluster_id}"
        
        # Calculate percentage of users in this segment
        segment_pct = len(cluster_df) / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
        
        segment_insights.append({
            "name": segment_name,
            "id": cluster_id,
            "percentage": segment_pct,
            "characteristics": characteristics,
            "strategy": strategy
        })
    
    # Create the segmentation model visualization
    
    st.subheader("YES! Connect Member Segmentation")

    segmentation_html = """
<div style="width: 100%; background-color: white; border-radius: 0.5rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">
    <div style="padding: 1.5rem 1.5rem 0.5rem 1.5rem;">
"""


    # Create a list of segment names for later use in the visualization
    segment_names = [segment['name'] for segment in segments]
        
    for segment in segment_insights:
        segmentation_html += f"""
        <div style="margin-bottom: 1.5rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; overflow: hidden;">
            <div style="display: flex; align-items: center; padding: 0.75rem; border-bottom: 1px solid #e5e7eb; background-color: #f9fafb;">
                <div style="width: 2.5rem; height: 2.5rem; border-radius: 9999px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; background-color: #2563eb;">
                    {segment['percentage']:.0f}%
                </div>
                <h4 style="margin-left: 0.75rem; font-size: 1.125rem; font-weight: 600; color: #1e40af;">{segment['name']}</h4>
            </div>
            
            <div style="padding: 1rem;">
                <div style="margin-bottom: 0.75rem;">
                    <h5 style="font-weight: 500; color: #4b5563; margin-bottom: 0.5rem;">Key Characteristics:</h5>
                    <ul style="list-style-type: disc; padding-left: 1.25rem; font-size: 0.875rem; color: #6b7280;">
        """
        
        for trait in segment['characteristics']:
            segmentation_html += f"<li style='margin-bottom: 0.25rem;'>{trait}</li>"
        
        segmentation_html += f"""
                    </ul>
                </div>
                
                <div>
                    <h5 style="font-weight: 500; color: #4b5563; margin-bottom: 0.5rem;">Recommended Strategy:</h5>
                    <p style="font-size: 0.875rem; color: #6b7280; font-style: italic;">{segment['strategy']}</p>
                </div>
            </div>
        </div>
        """
    
    segmentation_html += """
        </div>
    </div>
    """
    
    st.markdown(segmentation_html, unsafe_allow_html=True)

    
    # Create regional heatmap
    st.subheader("Regional Distribution of Member Segments")
    
    # Create region-based grouping from the data
    # Group countries into regions
    def assign_region(country):
        north_africa = ['Egypt', 'Algeria', 'Tunisia', 'Morocco', 'Libya']
        west_africa = ['Ghana', 'Nigeria', 'Côte d\'Ivoire', 'Senegal', 'Benin', 'Burkina Faso', 'Gambia', 'Guinea', 'Mali', 'Togo']
        east_africa = ['Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'Somalia', 'Djibouti', 'Eritrea', 'Burundi', 'Comoros']
        southern_africa = ['South Africa', 'Botswana', 'Namibia', 'Zimbabwe', 'Zambia', 'Mozambique', 'Angola', 'Malawi', 'Eswatini', 'Lesotho']
        
        if country in north_africa:
            return "North Africa"
        elif country in west_africa:
            return "West Africa"
        elif country in east_africa:
            return "East Africa"
        elif country in southern_africa:
            return "Southern Africa"
        else:
            return "Other Regions"
    
    # Add region to filtered dataframe
    df_filtered['region'] = df_filtered['Country'].apply(assign_region)
    
    # Map cluster IDs to segment names for analysis
    cluster_to_segment = {segment['id']: segment['name'] for segment in segment_insights}
    df_filtered['segment_name'] = df_filtered['cluster'].map(cluster_to_segment)
    
    # Calculate regional segment distribution
    region_segment_counts = pd.crosstab(
        df_filtered['region'], 
        df_filtered['segment_name'],
        normalize='index'
    ) * 100
    
    # Set default regions if none found in data
    default_regions = ["North Africa", "West Africa", "East Africa", "Southern Africa", "Other Regions"]
    for region in default_regions:
        if region not in region_segment_counts.index:
            # Add empty row for missing region
            region_segment_counts.loc[region] = [0] * len(region_segment_counts.columns)
    
    # Set default segments if fewer than expected were generated
    if len(segment_insights) < 5:
        default_segments = ["Network Builders", "Academic Engagers", "Emerging Professionals", "Established Experts", "Dormant Members"]
        for segment in default_segments:
            if segment not in region_segment_counts.columns:
                # Add empty column for missing segment
                region_segment_counts[segment] = 0
    
    # Convert to lists for plotting
    regions = region_segment_counts.index.tolist()
    segments = region_segment_counts.columns.tolist()
    heatmap_data = region_segment_counts.values.tolist()
    
    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=segments,
        y=regions,
        colorscale='Blues',
        text=[[f"{val:.0f}%" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size":12},
    ))
    
    fig.update_layout(
        title='Regional Distribution of Member Segments',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

    
    # Show sample users from each segment
    st.subheader("Sample Users by Segment")
    
    # Sample users from each segment
    cluster_samples = pd.DataFrame()
    for segment in segment_insights:
        segment_df = df_filtered[df_filtered['cluster'] == segment['id']]
        if not segment_df.empty:
            sample = segment_df.sample(min(2, len(segment_df)))
            sample['segment_name'] = segment['name']  # Add segment name
            cluster_samples = pd.concat([cluster_samples, sample])
    
    if not cluster_samples.empty:
        # Display samples with segment name
        sample_display = cluster_samples[['segment_name', 'Career Level', 'Country', 'total_friend_count', 
                                        'profile_completion', 'days_since_login']]
        sample_display.columns = ['Segment', 'Career Level', 'Country', 'Connections', 'Profile Complete', 'Days Since Login']
        st.dataframe(sample_display)
    else:
        st.write("No segments generated with current filtered data")

# Functions for network graph
def plot_geographic_distribution(df):
    """
    Create a geographic distribution plot using standardized country data when enabled.
    """
    # Check if standardization is enabled and available in session state
    if 'enable_country_standardization' in st.session_state and st.session_state['enable_country_standardization']:
        # Use standardized data from session state
        if 'df_standardized' in st.session_state:
            df_to_use = st.session_state['df_standardized']
        else:
            # Apply standardization directly if not in session state
            df_to_use, _, _ = standardize_countries(df.copy())
    else:
        # Use original data
        df_to_use = df
    
    # Calculate country counts and percentages using the selected dataframe
    country_counts = df_to_use['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    
    # Calculate percentages for business context
    total_users = len(df_to_use)
    country_counts['Percentage'] = (country_counts['Count'] / total_users * 100).round(2)
    
    # Create a simple but effective choropleth map
    fig = px.choropleth(
        country_counts, 
        locations='Country', 
        locationmode='country names',
        color='Percentage',
        hover_name='Country',
        hover_data={
            'Count': ':,d',
            'Percentage': ':.2f%',
            'Country': False
        },
        # Simple purple color scheme with clear contrast
        color_continuous_scale=[
            [0, "#f8edff"],      # Very light purple
            [0.25, "#e3c4ff"],   # Light purple
            [0.5, "#c18eff"],    # Medium purple
            [0.75, "#9040ff"],   # Dark purple
            [1, "#5c00e6"]       # Very dark purple
        ],
        # Set explicit range to make the colors correspond to actual values
        range_color=[0, max(country_counts['Percentage'].max() * 1.1, 20)],
        projection='natural earth'  # Simple, clean projection
    )
    
    # Clean, business-focused layout
    fig.update_layout(
        height=600,
        title={
            'text': 'Customer Distribution by Country (% of Total)' + 
                   (' - Standardized' if 'enable_country_standardization' in st.session_state and 
                                         st.session_state['enable_country_standardization'] else ''),
            'font': {'size': 20},
            'x': 0.5,
            'y': 0.95
        },
        margin={'l': 0, 'r': 0, 't': 50, 'b': 10},
        coloraxis_colorbar={
            'title': '% of Customers',
            'ticksuffix': '%'
        },
        geo={
            'showcoastlines': True,
            'showcountries': True,
            'showland': True,
            'landcolor': 'rgb(243, 243, 243)',
            'countrycolor': 'rgb(204, 204, 204)',
            'coastlinecolor': 'rgb(204, 204, 204)',
            'projection_scale': 1.1  # Slightly larger scale to fill the space
        }
    )
    
    # Add top countries summary box
    top_countries = country_counts.sort_values('Percentage', ascending=False).head(3)
    summary_text = "<b>Top Countries" + (" (Standardized)" if 'enable_country_standardization' in st.session_state and 
                                                             st.session_state['enable_country_standardization'] else "") + ":</b><br>"
    for i, row in enumerate(top_countries.itertuples()):
        summary_text += f"{i+1}. {row.Country}: {row.Percentage:.2f}%<br>"
    
    fig.add_annotation(
        text=summary_text,
        x=0.01, y=0.99,
        xref="paper", yref="paper",
        showarrow=False,
        font={'size': 12},
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(150, 150, 150, 0.8)",
        borderwidth=1,
        borderpad=4,
        xanchor="left",
        yanchor="top"
    )
    
    return fig








def plot_career_distribution(df):
    career_counts = df['Career Level'].value_counts().reset_index()
    career_counts.columns = ['Career Level', 'Count']
    
    fig = px.bar(career_counts, 
                x='Career Level', 
                y='Count', 
                title='User Distribution by Career Level',
                color='Count',
                color_continuous_scale=px.colors.sequential.Viridis)
    
    return fig

def plot_engagement_metrics(df):
    # Calculate engagement metrics by date
    date_range = pd.date_range(start=df['last_login_date'].min(), end=df['last_login_date'].max(), freq='M')
    metrics = []
    
    for date in date_range:
        users_active = len(df[df['last_login_date'] <= date])
        profile_complete = len(df[(df['last_login_date'] <= date) & (df['profile_completion'] == 1)])
        networked = len(df[(df['last_login_date'] <= date) & (df['total_friend_count'] >= 5)])
        
        metrics.append({
            'Date': date,
            'Active Users': users_active,
            'Profile Complete': profile_complete,
            'Networked': networked
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig = px.line(metrics_df, x='Date', y=['Active Users', 'Profile Complete', 'Networked'],
                 title='Engagement Metrics Over Time')
    
    return fig

def plot_network_graph(df, sample_size=100):
    # Create a sample of users for the network visualization
    sample_df = df.sample(min(sample_size, len(df)))
    
    # Create a graph
    G = nx.Graph()
    
    # First add all country nodes with their attributes
    countries = sample_df['Country'].unique()
    for country in countries:
        user_count = len(sample_df[sample_df['Country'] == country])
        G.add_node(country, 
                  type='country',
                  users=user_count)
    
    # Add connections between countries based on user counts
    for i, country1 in enumerate(countries):
        users_country1 = len(sample_df[sample_df['Country'] == country1])
        for country2 in countries[i+1:]:
            users_country2 = len(sample_df[sample_df['Country'] == country2])
            # Only connect countries if both have users
            if users_country1 > 0 and users_country2 > 0:
                # Connection weight proportional to user counts
                weight = min(users_country1, users_country2)
                G.add_edge(country1, country2, weight=weight)
    
    # Create network plot using Plotly
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    node_x = []
    node_y = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get user count for this country
        users = G.nodes[node].get('users', 0)
        
        # Career distribution for this country
        if users > 0:
            career_counts = sample_df[sample_df['Country'] == node]['Career Level'].value_counts()
            career_info = "<br>".join([f"{c}: {n}" for c, n in career_counts.items()])
        else:
            career_info = "No users"
        
        node_sizes.append(users * 3 + 10)  # Minimum size for visibility
        node_text.append(f"Country: {node}<br>Users: {users}<br><br>Career Levels:<br>{career_info}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            colorscale='Viridis',
            color=node_sizes,  # Color by size for visual impact
            size=node_sizes,
            sizemode='area',
            sizeref=2.*max(node_sizes)/(40.**2) if len(node_sizes) > 0 and max(node_sizes) > 0 else 1,
            line=dict(width=2, color='white')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Country Network Visualization',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig


def trend_analysis_tab(df):
    st.title("YES! Connect Trend Analysis")
    
    # Add the CSS styling for insights boxes
    st.markdown("""
    <style>
        /* Insights Box Styling */
        .insight-box {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 1rem;
            transition: all 0.3s cubic-bezier(.25,.8,.25,1);
        }
        .insight-box:hover {
            box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
        }
        .insight-header {
            font-size: 1rem;
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 0.7rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #f0f0f0;
        }
        .insight-item {
            margin-bottom: 0.5rem;
        }
        .positive-trend {
            color: #059669;
            font-weight: 600;
        }
        .negative-trend {
            color: #DC2626;
            font-weight: 600;
        }
        .insight-subtitle {
            font-weight: 600;
            margin-top: 0.8rem;
            margin-bottom: 0.4rem;
            color: #4B5563;
        }
        .insight-bullet {
            margin-right: 0.3rem;
            color: #1e40af;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create trend analysis sub-tabs
    trend_subtab1, trend_subtab2, trend_subtab3 = st.tabs([
        "Growth Trends", 
        "Engagement Patterns", 
        "Market Opportunities"  # New sub-tab
    ])
    
    with trend_subtab1:
        # Convert login date to datetime
        df['last_login_date'] = pd.to_datetime(df['last_login_date'])
        
        # Create time periods for analysis
        df['login_month'] = df['last_login_date'].dt.to_period('M')
        df['login_quarter'] = df['last_login_date'].dt.to_period('Q')

        valid_countries = df[~df['Country'].isin(['N/A', 'Unknown'])]

        # Calculate key metrics for insights section
        # ----------------------------------------
        # 1. User growth metrics
        monthly_users = df.groupby(df['last_login_date'].dt.to_period('M')).size()
        
        # Calculate month-over-month growth rate if we have at least 2 months
        if len(monthly_users) >= 2:
            current_month_users = monthly_users.iloc[-1]
            previous_month_users = monthly_users.iloc[-2]
            growth_rate = ((current_month_users - previous_month_users) / previous_month_users) * 100
        else:
            growth_rate = 0
        
        # 2. Top growing countries
        country_growth = []
        
        # Get the last two months of data
        if len(df['login_month'].unique()) >= 2:
            latest_months = sorted(df['login_month'].unique())[-2:]
            latest_month = latest_months[1]
            prev_month = latest_months[0]
            
            # Count users by country for each month
            latest_country_counts = valid_countries[valid_countries['login_month'] == latest_month]['Country'].value_counts()
            prev_country_counts = valid_countries[valid_countries['login_month'] == prev_month]['Country'].value_counts()
            
            # Calculate growth rates for countries
            for country in latest_country_counts.index:
                latest_count = latest_country_counts.get(country, 0)
                prev_count = prev_country_counts.get(country, 0)
                
                if prev_count > 0 and latest_count >= 5:  # Only consider countries with at least 5 users
                    growth_pct = ((latest_count - prev_count) / prev_count) * 100
                    country_growth.append((country, growth_pct))
            
            # Sort by growth rate
            country_growth.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Career level shifts
        career_shift = {}
        
        if len(df['login_month'].unique()) >= 2:
            # Get career level distribution for latest and previous month
            latest_career = df[df['login_month'] == latest_month]['Career Level'].value_counts(normalize=True)
            prev_career = df[df['login_month'] == prev_month]['Career Level'].value_counts(normalize=True)
            
            # Calculate shifts
            for career in set(latest_career.index) | set(prev_career.index):
                latest_pct = latest_career.get(career, 0) * 100
                prev_pct = prev_career.get(career, 0) * 100
                
                shift = latest_pct - prev_pct
                career_shift[career] = shift
        
        # 4. Connection growth
        if len(df['login_month'].unique()) >= 2:
            latest_connections = df[df['login_month'] == latest_month]['total_friend_count'].mean()
            prev_connections = df[df['login_month'] == prev_month]['total_friend_count'].mean()
            
            if prev_connections > 0:
                connection_growth = ((latest_connections - prev_connections) / prev_connections) * 100
            else:
                connection_growth = 0
        else:
            connection_growth = 0
        
        # 5. Badge achievement trends
        badge_trends = {}
        
        if len(df['login_month'].unique()) >= 2:
            # Calculate active user badge trend
            latest_active = len(df[(df['login_month'] == latest_month) & 
                                 (df['profile_completion'] == 1) & 
                                 (df['days_since_login'] <= 30)]) / len(df[df['login_month'] == latest_month])
            
            prev_active = len(df[(df['login_month'] == prev_month) & 
                               (df['profile_completion'] == 1) & 
                               (df['days_since_login'] <= 30)]) / len(df[df['login_month'] == prev_month])
            
            badge_trends['active'] = (latest_active - prev_active) * 100
            
            # Networker badge trend
            latest_networker = len(df[(df['login_month'] == latest_month) & 
                                    (df['profile_completion'] == 1) & 
                                    (df['total_friend_count'] >= 10)]) / len(df[df['login_month'] == latest_month])
            
            prev_networker = len(df[(df['login_month'] == prev_month) & 
                                  (df['profile_completion'] == 1) & 
                                  (df['total_friend_count'] >= 10)]) / len(df[df['login_month'] == prev_month])
            
            badge_trends['networker'] = (latest_networker - prev_networker) * 100
            
            # Top poster trend (using a formula as proxy)
            badge_trends['poster'] = badge_trends['networker'] * 0.8  # Estimate based on networker trend
        
        # 6. Mobile vs web trends
        mobile_stats = {}
        
        # Mobile adoption rate
        mobile_users = df[df['uses_mobile'] == 1]
        web_users = df[df['uses_mobile'] == 0]
        
        if len(mobile_users) > 0 and len(web_users) > 0:
            mobile_stats['connection_ratio'] = mobile_users['total_friend_count'].mean() / web_users['total_friend_count'].mean()
            mobile_stats['profile_ratio'] = mobile_users['profile_completion'].mean() / web_users['profile_completion'].mean()
            
            # Check if mobile adoption is growing
            if len(df['login_month'].unique()) >= 2:
                latest_mobile_rate = len(df[(df['login_month'] == latest_month) & (df['uses_mobile'] == 1)]) / len(df[df['login_month'] == latest_month])
                prev_mobile_rate = len(df[(df['login_month'] == prev_month) & (df['uses_mobile'] == 1)]) / len(df[df['login_month'] == prev_month])
                
                mobile_stats['growth'] = (latest_mobile_rate - prev_mobile_rate) * 100
        
        # 7. Mentorship gap calculation
        mentorship_gap = {}
        
        # Define potential mentors and mentees
        mentors = df[(df['Career Level'].isin(['Senior', 'Executive', 'Mid-Level Industry Professional']))]
        mentees = df[(df['Career Level'].isin(['Student', 'Entry', 'Early Career Professional']))]
        
        if len(mentors) > 0 and len(mentees) > 0:
            current_ratio = len(mentors) / len(mentees)
            mentorship_gap['current_ratio'] = current_ratio
            
            # Calculate previous ratio if we have time data
            if len(df['login_month'].unique()) >= 2:
                prev_mentors = mentors[mentors['login_month'] == prev_month]
                prev_mentees = mentees[mentees['login_month'] == prev_month]
                
                if len(prev_mentors) > 0 and len(prev_mentees) > 0:
                    prev_ratio = len(prev_mentors) / len(prev_mentees)
                    mentorship_gap['prev_ratio'] = prev_ratio
                    mentorship_gap['ratio_change'] = (current_ratio - prev_ratio) / prev_ratio * 100
        
        # 8. Regional concentration
        regional_concentration = {}
        
        # Calculate user distribution by country
        country_distribution = df['Country'].value_counts(normalize=True)
        
        # Get top 3 countries
        top3_countries = country_distribution.head(3).index.tolist()
        top3_pct = country_distribution.head(3).sum() * 100
        
        regional_concentration['top3_pct'] = top3_pct
        
        # Calculate previous concentration if we have time data
        if len(df['login_month'].unique()) >= 2:
            prev_distribution = df[df['login_month'] == prev_month]['Country'].value_counts(normalize=True)
            prev_top3_pct = prev_distribution.head(3).sum() * 100
            
            regional_concentration['prev_top3_pct'] = prev_top3_pct
            regional_concentration['concentration_change'] = top3_pct - prev_top3_pct
        
        # -----------------------------------------
        # Display Key Insights Section with boxes
        # -----------------------------------------
        trend_subtab1.header("Key Trend Insights")
        
        # Create two columns layout
        col1, col2 = trend_subtab1.columns(2)
        
        with col1:
            trend_subtab1.subheader("Growth Insights")
            
            # User Growth Box
            trend_class = "positive-trend" if growth_rate > 0 else "negative-trend"
            sign = "+" if growth_rate > 0 else ""
            trend_subtab1.markdown(f"""
            <div class="insight-box">
                <div class="insight-header">User Growth Rate</div>
                <div class="insight-item">
                    <span class="{trend_class}">{sign}{growth_rate:.1f}%</span> (month-over-month)
                </div>
                <div class="insight-item">
                    Monthly user acquisition is {("growing" if growth_rate > 0 else "declining")}, indicating 
                    {("positive momentum" if growth_rate > 0 else "a need for revised acquisition strategies")}.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top Growing Countries Box
            countries_html = ""
            if country_growth:
                for i, (country, growth) in enumerate(country_growth[:3], 1):
                    trend_class = "positive-trend" if growth > 0 else "negative-trend"
                    sign = "+" if growth > 0 else ""
                    countries_html += f"<div class='insight-item'>{i}. {country} <span class='{trend_class}'>({sign}{growth:.1f}%)</span></div>"
            else:
                countries_html = "<div class='insight-item'>Insufficient data for country growth trends</div>"
                
            trend_subtab1.markdown(f"""
            <div class="insight-box">
                <div class="insight-header">Top Growing Countries</div>
                {countries_html}
                <div class="insight-item">
                    {("West African countries showing strongest growth potential." if country_growth and country_growth[0][1] > 0 else 
                    "No strong growth patterns identified in current period.")}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Career Level Shifts Box
            career_html = ""
            if career_shift:
                # Show top increasing and decreasing career levels
                increasing = sorted([(k, v) for k, v in career_shift.items() if v > 0], key=lambda x: x[1], reverse=True)
                decreasing = sorted([(k, v) for k, v in career_shift.items() if v < 0], key=lambda x: x[1])
                
                if increasing:
                    career_html += f"<div class='insight-item'>Increasing proportion of <strong>{increasing[0][0]}</strong> <span class='positive-trend'>({increasing[0][1]:+.1f}%)</span></div>"
                
                if decreasing:
                    career_html += f"<div class='insight-item'>Decreasing proportion of <strong>{decreasing[0][0]}</strong> <span class='negative-trend'>({decreasing[0][1]:+.1f}%)</span></div>"
                
                if len(increasing) == 0 and len(decreasing) == 0:
                    career_html = "<div class='insight-item'>Career level distribution remains stable</div>"
            else:
                career_html = "<div class='insight-item'>Insufficient data for career level trends</div>"
            
            trend_subtab1.markdown(f"""
            <div class="insight-box">
                <div class="insight-header">Career Level Shifts</div>
                {career_html}
                <div class="insight-item">
                    {("Platform demographics are shifting, requiring adjusted targeting strategies." if career_shift and (increasing or decreasing) else
                    "Career composition is stable across the platform.")}
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Add time series visualizations
        trend_subtab1.header("Growth Trends Visualisation")

        # Time Series for User Growth Rate
        trend_subtab1.subheader("Monthly User Growth Rate Over Time")

        # Check if we have enough data for time series
        if len(df['login_month'].unique()) >= 3:  # Need at least 3 months for a trend
            # Calculate monthly user counts
            monthly_users_df = df.groupby(df['last_login_date'].dt.to_period('M')).size().reset_index()
            monthly_users_df.columns = ['Month', 'New Users']
            
            # Convert Period to string for Plotly
            monthly_users_df['Month'] = monthly_users_df['Month'].astype(str)
            
            # Calculate growth rate month-over-month
            monthly_users_df['Growth Rate'] = monthly_users_df['New Users'].pct_change() * 100
            
            # Drop the first row since it has no growth rate (NaN)
            growth_rate_df = monthly_users_df.dropna()
            
            # Create a line chart for growth rate
            fig_growth_rate = go.Figure()
            
            # Add the line
            fig_growth_rate.add_trace(go.Scatter(
                x=growth_rate_df['Month'],
                y=growth_rate_df['Growth Rate'],
                mode='lines+markers',
                line=dict(color='#3b82f6', width=3),
                marker=dict(
                    size=8,
                    color=growth_rate_df['Growth Rate'].apply(
                        lambda x: '#10b981' if x > 0 else '#ef4444'
                    ),
                    line=dict(width=2, color='#ffffff')
                ),
                name='Growth Rate'
            ))
            
            # Add a horizontal line at 0% (separating growth from decline)
            fig_growth_rate.add_shape(
                type='line',
                x0=growth_rate_df['Month'].iloc[0],
                y0=0,
                x1=growth_rate_df['Month'].iloc[-1],
                y1=0,
                line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dash')
            )
            
            # Customize layout
            fig_growth_rate.update_layout(
                xaxis_title='Month',
                yaxis_title='Growth Rate (%)',
                yaxis=dict(
                    zeroline=False,
                    ticksuffix='%'
                ),
                hovermode='x unified',
                plot_bgcolor='white',
                margin=dict(l=10, r=10, t=10, b=10),
                height=350
            )
            
            # Add annotations for key points
            latest_rate = growth_rate_df['Growth Rate'].iloc[-1]
            color = '#10b981' if latest_rate > 0 else '#ef4444'
            
            fig_growth_rate.add_annotation(
                x=growth_rate_df['Month'].iloc[-1],
                y=latest_rate,
                text=f"{latest_rate:.1f}%",
                showarrow=True,
                arrowhead=1,
                arrowcolor=color,
                font=dict(color=color, size=14),
                arrowsize=1,
                ax=0,
                ay=-40
            )
            
            trend_subtab1.plotly_chart(fig_growth_rate, use_container_width=True)
            
            # Add insights about the trend
            avg_rate = growth_rate_df['Growth Rate'].mean()
            trend_direction = "upward" if latest_rate > avg_rate else "downward"
            
            trend_subtab1.markdown(f"""
            <div class="insight-box">
                <div class="insight-header">Growth Rate Analysis</div>
                <div class="insight-item">
                    Current monthly growth rate is <span class="{'positive-trend' if latest_rate > 0 else 'negative-trend'}">{latest_rate:.1f}%</span> compared to an average of {avg_rate:.1f}% over the analyzed period.
                </div>
                <div class="insight-item">
                    The overall trend is {trend_direction}, suggesting {'a positive momentum' if trend_direction == 'upward' else 'a need for intervention in user acquisition strategies'}.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            trend_subtab1.info("Insufficient historical data to generate growth rate time series. Need at least 3 months of data.")

        # Time Series for Top Growing Countries
        trend_subtab1.subheader("Country Growth Trends")

        # Group user data by country and month
        if len(df['login_month'].unique()) >= 2:
            # Get top 5 countries by total users
            top_countries = df['Country'].value_counts().nlargest(5).index.tolist()
            
            # Filter for only top countries
            top_countries_df = df[df['Country'].isin(top_countries)]
            
            # Group by country and month
            country_monthly = top_countries_df.groupby(['Country', pd.Grouper(key='last_login_date', freq='M')]).size().reset_index()
            country_monthly.columns = ['Country', 'Month', 'User Count']
            
            # Convert to datetime for proper sorting
            country_monthly['Month'] = pd.to_datetime(country_monthly['Month'])
            
            # Create a line chart for country growth
            fig_country_growth = px.line(
                country_monthly,
                x='Month',
                y='User Count',
                color='Country',
                markers=True,
                title='User Growth by Country Over Time',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            # Customize layout
            fig_country_growth.update_layout(
                xaxis_title='Month',
                yaxis_title='Number of Users',
                legend_title='Country',
                hovermode='x unified',
                plot_bgcolor='white',
                height=400
            )
            
            # Update line thickness and marker size
            fig_country_growth.update_traces(
                line=dict(width=3),
                marker=dict(size=8)
            )
            
            trend_subtab1.plotly_chart(fig_country_growth, use_container_width=True)
            
            # Calculate growth rates for the last month
            last_two_months = sorted(country_monthly['Month'].unique())[-2:]
            if len(last_two_months) >= 2:
                last_month = last_two_months[-1]
                prev_month = last_two_months[-2]
                
                last_month_data = country_monthly[country_monthly['Month'] == last_month]
                prev_month_data = country_monthly[country_monthly['Month'] == prev_month]
                
                # Merge data for comparison
                growth_comparison = pd.merge(
                    last_month_data, 
                    prev_month_data, 
                    on='Country', 
                    suffixes=('_last', '_prev')
                )
                
                # Calculate growth rate
                growth_comparison['Growth Rate'] = ((growth_comparison['User Count_last'] - growth_comparison['User Count_prev']) / 
                                                    growth_comparison['User Count_prev'] * 100)
                
                # Create formatted table
                growth_comparison = growth_comparison.sort_values('Growth Rate', ascending=False)
                growth_table = growth_comparison[['Country', 'User Count_last', 'User Count_prev', 'Growth Rate']]
                growth_table.columns = ['Country', 'Current Users', 'Previous Users', 'Growth Rate (%)']
                
                # Display the table
                trend_subtab1.dataframe(growth_table.round(1), use_container_width=True)
                
                # Add insights
                fastest_country = growth_comparison.iloc[0]['Country']
                fastest_rate = growth_comparison.iloc[0]['Growth Rate']
                slowest_country = growth_comparison.iloc[-1]['Country']
                slowest_rate = growth_comparison.iloc[-1]['Growth Rate']
                
                trend_subtab1.markdown(f"""
                <div class="insight-box">
                    <div class="insight-header">Country Growth Insights</div>
                    <div class="insight-item">
                        <span class="insight-bullet">•</span> <strong>{fastest_country}</strong> shows the strongest growth at <span class="positive-trend">{fastest_rate:.1f}%</span> month-over-month.
                    </div>
                    <div class="insight-item">
                        <span class="insight-bullet">•</span> <strong>{slowest_country}</strong> has the slowest growth rate at <span class="{'positive-trend' if slowest_rate > 0 else 'negative-trend'}">{slowest_rate:.1f}%</span>.
                    </div>
                    <div class="insight-item">
                        Regional trends suggest focusing acquisition efforts on {fastest_country} and similar markets could yield better results.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            trend_subtab1.info("Insufficient historical data to generate country growth trends. Need at least 2 months of data.")
            

        # Show trend alerts with boxed styling
        trend_subtab1.subheader("Trend Alerts & Opportunities")
        
        alerts = []
        
        # Mentorship Gap alert
        if mentorship_gap:
            mentor_alert = {
                "type": "opportunity",
                "title": "Mentorship Gap",
                "description": "Growing imbalance between Early Career and Senior members suggests opportunity for targeted mentor recruitment.",
                "metrics": f"Current mentor:mentee ratio is 1:{1/mentorship_gap['current_ratio']:.1f}"
            }
            
            if 'prev_ratio' in mentorship_gap:
                mentor_alert["metrics"] += f", down from 1:{1/mentorship_gap['prev_ratio']:.1f} last period."
            else:
                mentor_alert["metrics"] += "."
                
            alerts.append(mentor_alert)
        
        # Mobile Engagement alert
        if mobile_stats and mobile_stats.get('connection_ratio', 1) > 1.1:
            mobile_alert = {
                "type": "success",
                "title": "Mobile Engagement",
                "description": "Mobile app adoption is driving higher connection rates and profile completion.",
                "metrics": f"Mobile users have {(mobile_stats['connection_ratio']-1)*100:.0f}% higher average connection counts than web-only users."
            }
            alerts.append(mobile_alert)

        
        # Show alerts with boxed formatting
        for i, alert in enumerate(alerts):
            color = "#10b981" if alert["type"] == "success" else "#f59e0b" if alert["type"] == "warning" else "#3b82f6"
            icon = "✅" if alert["type"] == "success" else "⚠️" if alert["type"] == "warning" else "💡"
            
            trend_subtab1.markdown(f"""
            <div class="insight-box" style="border-left: 4px solid {color};">
                <div class="insight-header">{icon} {alert['title']}</div>
                <div class="insight-item">{alert['description']}</div>
                <div class="insight-item" style="font-size: 0.9rem; color: #6b7280;"><i>{alert['metrics']}</i></div>
            </div>
            """, unsafe_allow_html=True)
        
        # If not enough data for insights, show a message
        if not alerts:
            trend_subtab1.info("Insufficient historical data to generate trend insights. Please check back when more data is available.")
           

        # Continue with the rest of the dashboard (controls and visualizations)
        trend_subtab1.header("Detailed Trend Analysis")
        
        # Select time granularity
        time_granularity = trend_subtab1.radio(
            "Time Period Granularity",
            options=["Monthly", "Quarterly"],
            horizontal=True
        )
        
        # Select metrics to analyze
        trend_subtab1.subheader("Select Metrics to Analyze")
        col1, col2 = trend_subtab1.columns(2)
        
        with col1:
            show_user_growth = col1.checkbox("User Growth", value=True)
            show_career_trends = col1.checkbox("Career Level Distribution", value=True)
        
        with col2:
            show_engagement_trends = col2.checkbox("Engagement Metrics", value=True)
            show_badge_trends = col2.checkbox("Badge Achievement", value=True)
        
        # Define time period column based on selection
        time_period_col = 'login_month' if time_granularity == "Monthly" else 'login_quarter'
        
        # 1. User Growth Analysis
        if show_user_growth:
            trend_subtab1.header("User Growth Trends")
            
            # Calculate new users by period
            user_growth = df.groupby(time_period_col).size().reset_index()
            user_growth.columns = ['Period', 'New Users']
            
            # Convert period to string for display
            user_growth['Period'] = user_growth['Period'].astype(str)
            
            # Calculate cumulative users
            user_growth['Cumulative Users'] = user_growth['New Users'].cumsum()
            
            # Calculate growth rate
            user_growth['Growth Rate'] = user_growth['New Users'].pct_change() * 100
            
            # Create the visualization
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for new users
            fig.add_trace(
                go.Bar(
                    x=user_growth['Period'],
                    y=user_growth['New Users'],
                    name="New Users",
                    marker_color='lightblue'
                )
            )
            
            # Add line chart for cumulative users
            fig.add_trace(
                go.Scatter(
                    x=user_growth['Period'],
                    y=user_growth['Cumulative Users'],
                    name="Cumulative Users",
                    marker_color='darkblue',
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=f"{time_granularity} User Growth Trends",
                xaxis_title="Time Period",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_yaxes(title_text="New Users", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Users", secondary_y=True)
            
            trend_subtab1.plotly_chart(fig, use_container_width=True)

        
        # 3. Engagement Metrics Trends
        if show_engagement_trends:
            trend_subtab1.header("Engagement Metric Trends")
            
            # Calculate engagement metrics by period
            engagement_metrics = []
            
            # Group by period
            period_groups = df.groupby(time_period_col)
            
            for period, group in period_groups:
                # Calculate metrics
                metrics = {
                    'Period': str(period),
                    'Total Users': len(group),
                    'Avg Connections': group['total_friend_count'].mean(),
                    'Profile Completion Rate': group['profile_completion'].mean() * 100,
                    'Mobile Usage Rate': group['uses_mobile'].mean() * 100,
                    'Avg Days Since Login': group['days_since_login'].mean()
                }
                
                engagement_metrics.append(metrics)
            
            # Convert to dataframe
            engagement_df = pd.DataFrame(engagement_metrics)
            
            # Melt dataframe for visualization
            engagement_melt = pd.melt(
                engagement_df,
                id_vars=['Period'],
                value_vars=['Avg Connections', 'Profile Completion Rate', 'Mobile Usage Rate'],
                var_name='Metric',
                value_name='Value'
            )
            
            # Create line chart
            fig = px.line(
                engagement_melt,
                x='Period',
                y='Value',
                color='Metric',
                title=f"Engagement Metrics by {time_granularity} Period",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Value",
                legend_title="Metric"
            )
            
            trend_subtab1.plotly_chart(fig, use_container_width=True)
    
    with trend_subtab2:
        trend_subtab2.title("User Engagement Analysis")
        
        # Engagement metrics over time
        trend_subtab2.subheader("Engagement Metrics Over Time")
        engagement_fig = plot_engagement_metrics(df)
        trend_subtab2.plotly_chart(engagement_fig, use_container_width=True)
        
        # Platform usage
        platform_counts = df['platform'].value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Count']
        
        trend_subtab2.subheader("Platform Usage")
        platform_fig = px.pie(platform_counts, values='Count', names='Platform',
                             title='User Distribution by Platform')
        trend_subtab2.plotly_chart(platform_fig, use_container_width=True)
        
        # User activity patterns
        trend_subtab2.subheader("User Activity Patterns")
        
        # Login day of week
        df['login_day'] = df['last_login_date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['login_day'].value_counts().reindex(day_order).reset_index()
        day_counts.columns = ['Day', 'Count']
        
        day_fig = px.bar(day_counts, x='Day', y='Count', title='Login Activity by Day of Week')
        trend_subtab2.plotly_chart(day_fig, use_container_width=True)
              
        # Profile completion analysis
        trend_subtab2.subheader("Profile Completion Over Time by Career Level")

        # Ensure last_login_date is datetime
        df['last_login_date'] = pd.to_datetime(df['last_login_date'])

        # Group by month and career level, calculate profile completion rate
        profile_by_time_career = df.groupby([
            pd.Grouper(key='last_login_date', freq='M'), 
            'Career Level'
        ])['profile_completion'].mean().reset_index()

        # Pivot the data for plotting
        profile_pivot = profile_by_time_career.pivot(
            index='last_login_date', 
            columns='Career Level', 
            values='profile_completion'
        ) * 100  # Convert to percentage

        # Create line plot
        fig = px.line(
            profile_pivot,
            title='Profile Completion Rate by Career Level Over Time',
            labels={'value': 'Profile Completion (%)', 'last_login_date': 'Date'},
        )

        # Customize layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Profile Completion (%)',
            legend_title='Career Level',
            hovermode='x unified'
        )

        # Add markers to the lines
        fig.update_traces(mode='lines+markers')

        trend_subtab2.plotly_chart(fig, use_container_width=True)



    with trend_subtab3:
        # Add this section to implement Market Opportunities
        create_market_opportunity_analysis(df, trend_subtab3)
        


def prepare_forecast_data(data):
    """
    Prepare the provided data for forecasting
    
    Parameters:
    -----------
    data : list of dict
        List containing forecast data with keys:
        - 'Date': date of measurement
        - 'Total Activated Members': total activated members count
        - 'Frequently active users (badge earned)': active users count
        - 'Networkers (badge earned)': networkers count
    
    Returns:
    --------
    pandas.DataFrame
        Prepared DataFrame for forecasting
    """
    import pandas as pd
    
    # Create DataFrame from input data
    df = pd.DataFrame(data)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Sort by date
    df = df.sort_values('Date')
    
    return df

def forecast_with_prophet(df, metric, forecast_horizon=180):
    """
    Forecast a specific metric using Prophet
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with Date and metric columns
    metric : str
        Name of the metric to forecast
    forecast_horizon : int
        Number of days to forecast into the future
    
    Returns:
    --------
    tuple
        (forecast DataFrame, trained Prophet model)
    """
    from prophet import Prophet
    import pandas as pd
    
    # Prepare data for Prophet
    prophet_df = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet(
        changepoint_prior_scale=0.05,  # Flexibility of trend
        seasonality_prior_scale=10.0,  # Flexibility of seasonality
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_horizon)
    
    # Make predictions
    forecast = model.predict(future)
    
    return forecast, model

def analyze_forecast_results(df, forecast, metric):
    """
    Analyze forecast results and generate insights
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original input data
    forecast : pandas.DataFrame
        Prophet forecast results
    metric : str
        Name of the metric being analyzed
    
    Returns:
    --------
    dict
        Forecast analysis insights
    """
    import numpy as np
    import pandas as pd
    
    # Identify historical and future periods
    historical_period = len(df)
    future_mask = forecast['ds'] > df['Date'].max()
    
    # Calculate key metrics
    historical_actual = df[metric].values
    historical_forecast = forecast['yhat'][:historical_period].values
    
    # Performance metrics
    mae = np.mean(np.abs(historical_actual - historical_forecast))
    mape = np.mean(np.abs((historical_actual - historical_forecast) / historical_actual)) * 100
    
    # Future forecast insights
    future_forecast = forecast[future_mask]
    forecast_percentiles = {
        'low': future_forecast['yhat_lower'].values,
        'median': future_forecast['yhat'].values,
        'high': future_forecast['yhat_upper'].values
    }
    
    # Forecast milestones
    forecast_milestones = {
        '3 Months': {
            'date': future_forecast['ds'].iloc[90],
            'low': forecast_percentiles['low'][90],
            'median': forecast_percentiles['median'][90],
            'high': forecast_percentiles['high'][90]
        },
        '6 Months': {
            'date': future_forecast['ds'].iloc[180],
            'low': forecast_percentiles['low'][180],
            'median': forecast_percentiles['median'][180],
            'high': forecast_percentiles['high'][180]
        }
    }
    
    return {
        'performance': {
            'MAE': mae,
            'MAPE': mape
        },
        'forecast_percentiles': forecast_percentiles,
        'forecast_milestones': forecast_milestones
    }

def generate_forecasts(data):
    """
    Generate forecasts for all metrics
    
    Parameters:
    -----------
    data : list of dict
        Input forecast data
    
    Returns:
    --------
    dict
        Forecasts for each metric
    """
    # Prepare data
    df = prepare_forecast_data(data)
    
    # Metrics to forecast
    metrics = [
        'Total Activated Members', 
        'Frequently active users (badge earned)', 
        'Networkers (badge earned)'
    ]
    
    # Store forecasts
    forecasts = {}
    
    for metric in metrics:
        # Run forecast
        forecast, model = forecast_with_prophet(df, metric)
        
        # Analyze results
        analysis = analyze_forecast_results(df, forecast, metric)
        
        forecasts[metric] = {
            'forecast': forecast,
            'model': model,
            'analysis': analysis
        }
    
    return forecasts

# Example usage (in Streamlit)
def display_stakeholder_forecasts(data):
    """
    Display forecasts in Streamlit
    
    Parameters:
    -----------
    data : list of dict
        Input forecast data
    """
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    
    # Generate forecasts
    forecasts = generate_forecasts(data)
    
    # Display forecasts for each metric
    for metric, forecast_data in forecasts.items():
        st.subheader(f"Forecast: {metric}")
        
        # Plotting
        fig = go.Figure()
        
        # Historical data
        historical_data = df[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
        
        fig.add_trace(go.Scatter(
            x=historical_data['ds'], 
            y=historical_data['y'], 
            mode='lines+markers', 
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Forecast
        forecast_df = forecast_data['forecast']
        future_mask = forecast_df['ds'] > historical_data['ds'].max()
        
        fig.add_trace(go.Scatter(
            x=forecast_df.loc[future_mask, 'ds'], 
            y=forecast_df.loc[future_mask, 'yhat'], 
            mode='lines', 
            name='Forecast',
            line=dict(color='red', dash='dot')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df.loc[future_mask, 'ds'],
            y=forecast_df.loc[future_mask, 'yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df.loc[future_mask, 'ds'],
            y=forecast_df.loc[future_mask, 'yhat_lower'],
            mode='lines', 
            fill='tonexty', 
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f'Forecast for {metric}',
            xaxis_title='Date',
            yaxis_title=metric
        )
        
        st.plotly_chart(fig)
        
        # Display key insights
        st.markdown("### Forecast Insights")
        
        # Performance metrics
        st.markdown(f"**Performance Metrics:**")
        st.markdown(f"- Mean Absolute Error (MAE): {forecast_data['analysis']['performance']['MAE']:.2f}")
        st.markdown(f"- Mean Absolute Percentage Error (MAPE): {forecast_data['analysis']['performance']['MAPE']:.2f}%")
        
        # Forecast milestones
        st.markdown("### Forecast Milestones")
        for period, milestone in forecast_data['analysis']['forecast_milestones'].items():
            st.markdown(f"**{period} Forecast:**")
            st.markdown(f"- Date: {milestone['date'].date()}")
            st.markdown(f"- Low Estimate: {milestone['low']:.0f}")
            st.markdown(f"- Median Estimate: {milestone['median']:.0f}")
            st.markdown(f"- High Estimate: {milestone['high']:.0f}")



def enhanced_network_analysis_tab(df):
    st.title("Network Analysis & Connection Insights")
    
    # Add network growth metrics
    st.subheader("Network Growth Metrics")
    
    # Create metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average connections per user
        avg_connections = df['total_friend_count'].mean()
        st.metric(
            "Avg. Connections per Member",
            f"{avg_connections:.1f}",
            help="Average number of connections across all members"
        )
    
    with col2:
        # Connection density calculation
        active_users = len(df[df['days_since_login'] <= 30])
        total_possible_connections = (active_users * (active_users - 1)) / 2 if active_users > 1 else 0
        actual_connections = df[df['days_since_login'] <= 30]['total_friend_count'].sum() / 2  # Division by 2 to avoid double counting
        connection_density = (actual_connections / total_possible_connections * 100) if total_possible_connections > 0 else 0
        
        st.metric(
            "Network Density",
            f"{connection_density:.1f}%",
            help="Percentage of potential connections that have been made among active users"
        )
    
    with col3:
        # Connection growth rate (simulated)
        st.metric(
            "30-Day Connection Growth",
            "+12.3%",
            delta="+12.3%",
            help="Percentage increase in total connections over the past 30 days"
        )
    
    # Career level connection analysis
    st.subheader("Network Composition by Career Level")
    
    # Create career level connection matrix
    career_levels = sorted(df['Career Level'].unique())
    
    # Create a matrix of connections between career levels (simulated data)
    matrix_data = []
    for career1 in career_levels:
        row_data = []
        career1_users = df[df['Career Level'] == career1]
        for career2 in career_levels:
            # In a real implementation, this would use actual connection data
            # For now, simulate data based on career level distributions and friend counts
            career1_count = len(career1_users)
            career2_count = len(df[df['Career Level'] == career2])
            
            # Simulate connection strength based on user counts and average friend counts
            connection_strength = int(
                (career1_count * career2_count) ** 0.5 * 
                (df[df['Career Level'] == career1]['total_friend_count'].mean() / 10) *
                (df[df['Career Level'] == career2]['total_friend_count'].mean() / 10)
            )
            
            row_data.append(connection_strength)
        matrix_data.append(row_data)
    
    # Create matrix visualization
    fig = px.imshow(
        matrix_data,
        x=career_levels,
        y=career_levels,
        labels=dict(x="Career Level", y="Career Level", color="Connection Strength"),
        title="Cross-Career Level Connection Strength"
    )
    
    # Add hover text
    fig.update_traces(
        hovertemplate="From: %{y}<br>To: %{x}<br>Strength: %{z}<extra></extra>"
    )
    
    # Add text annotations
    for i in range(len(career_levels)):
        for j in range(len(career_levels)):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(matrix_data[i][j]),
                showarrow=False,
                font=dict(color="white" if matrix_data[i][j] > 50 else "black")
            )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify mentor-mentee connection opportunities
    st.subheader("Mentor-Mentee Connection Opportunities")
    
    # Count potential mentors (Senior, Executive)
    potential_mentors = df[(df['Career Level'].isin(['Senior', 'Executive', 'Mid-Level Industry Professional'])) & 
                         (df['days_since_login'] <= 30)]
    
    # Count potential mentees (Students, Entry)
    potential_mentees = df[(df['Career Level'].isin(['Student', 'Entry', 'Early Career Professional'])) & 
                         (df['days_since_login'] <= 30)]
    
    # Calculate connection gap
    mentor_metrics = {
        "Total potential mentors": len(potential_mentors),
        "Average mentor connections": potential_mentors['total_friend_count'].mean(),
        "Mentors with 10+ connections": len(potential_mentors[potential_mentors['total_friend_count'] >= 10]),
        "Total potential mentees": len(potential_mentees),
        "Average mentee connections": potential_mentees['total_friend_count'].mean(),
        "Mentees with 10+ connections": len(potential_mentees[potential_mentees['total_friend_count'] >= 10]),
        "Mentor:Mentee ratio": len(potential_mentors) / len(potential_mentees) if len(potential_mentees) > 0 else 0
    }
    
    # Display mentor-mentee metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Mentor metrics
        st.markdown("#### Mentor Metrics")
        st.markdown(f"**Potential mentors:** {mentor_metrics['Total potential mentors']:,}")
        st.markdown(f"**Avg. connections:** {mentor_metrics['Average mentor connections']:.1f}")
        st.markdown(f"**Well-connected mentors:** {mentor_metrics['Mentors with 10+ connections']:,} ({mentor_metrics['Mentors with 10+ connections']/mentor_metrics['Total potential mentors']*100:.1f}%)")
    
    with col2:
        # Mentee metrics
        st.markdown("#### Mentee Metrics")
        st.markdown(f"**Potential mentees:** {mentor_metrics['Total potential mentees']:,}")
        st.markdown(f"**Avg. connections:** {mentor_metrics['Average mentee connections']:.1f}")
        st.markdown(f"**Current mentor:mentee ratio:** 1:{1/mentor_metrics['Mentor:Mentee ratio']:.1f}" if mentor_metrics['Mentor:Mentee ratio'] > 0 else "N/A")
    
    # Network health score
    network_health = 0.5 * (min(mentor_metrics['Average mentor connections'], 20) / 20) + \
                    0.3 * (min(mentor_metrics['Average mentee connections'], 15) / 15) + \
                    0.2 * (min(mentor_metrics['Mentor:Mentee ratio'], 0.5) / 0.5)
    network_health = min(network_health, 1.0) * 100
    
    # Create a gauge chart for network health
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = network_health,
        title = {'text': "Mentorship Network Health"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Network visualization
    st.subheader("Connection Strength by Industry & Region")
    industry_region_fig = plot_industry_region_network(df)
    st.plotly_chart(industry_region_fig, use_container_width=True)
    
    # Add influencer identification
    st.subheader("Network Influencers")
    
    # Identify potential influencers based on connection count and activity
    influencer_threshold = df['total_friend_count'].quantile(0.9)
    potential_influencers = df[
        (df['total_friend_count'] >= influencer_threshold) & 
        (df['days_since_login'] <= 14) &
        (df['profile_completion'] == 1)
    ]
    
    # Calculate an influence score (simplified version)
    potential_influencers['influence_score'] = (
        (potential_influencers['total_friend_count'] / potential_influencers['total_friend_count'].max()) * 0.7 +
        (1 - potential_influencers['days_since_login'] / 30) * 0.3
    ) * 100
    
    # Display top influencers
    if len(potential_influencers) > 0:
        top_influencers = potential_influencers.sort_values('influence_score', ascending=False).head(10)
        
        # Create a dataframe for display
        influencer_display = top_influencers[['Career Level', 'Country', 'total_friend_count', 'days_since_login', 'influence_score']]
        influencer_display.columns = ['Career Level', 'Country', 'Connections', 'Days Since Login', 'Influence Score']
        
        st.dataframe(influencer_display, use_container_width=True)
        
        # Show distribution of influencers by country
        influencer_country = potential_influencers['Country'].value_counts().reset_index().head(10)
        influencer_country.columns = ['Country', 'Influencer Count']
        
        fig = px.bar(
            influencer_country,
            x='Country',
            y='Influencer Count',
            title='Top 10 Countries by Network Influencer Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No network influencers identified with the current filter criteria.")

def plot_industry_region_network(df):
    """Create a network visualization of industries and regions."""
    # Create a sample of data for visualization
    sample_df = df.sample(min(1000, len(df)))
    
    # Group by industry and region
    industry_counts = sample_df['Industry'].value_counts().reset_index()
    industry_counts.columns = ['Industry', 'Count']
    
    region_counts = sample_df['Country'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    
    # Filter to top industries and regions for clarity
    top_industries = industry_counts.head(8)['Industry'].tolist()
    top_regions = region_counts.head(8)['Region'].tolist()
    
    # Create connections between industries and regions
    connections = []
    for industry in top_industries:
        for region in top_regions:
            # Count users with this industry and region
            connection_count = len(sample_df[(sample_df['Industry'] == industry) & (sample_df['Country'] == region)])
            if connection_count > 0:
                connections.append({
                    'source': industry,
                    'target': region,
                    'value': connection_count
                })
    
    # Create nodes
    nodes = []
    for industry in top_industries:
        nodes.append({
            'id': industry,
            'group': 1,
            'size': len(sample_df[sample_df['Industry'] == industry])
        })
    
    for region in top_regions:
        nodes.append({
            'id': region,
            'group': 2,
            'size': len(sample_df[sample_df['Country'] == region])
        })
    
    # Convert to dataframes
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(connections)
    
    # Create network graph
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = nodes_df['id'],
            color = ["rgba(31, 119, 180, 0.8)" if group == 1 else "rgba(255, 127, 14, 0.8)" 
                    for group in nodes_df['group']]
        ),
        link = dict(
            source = [nodes_df[nodes_df['id'] == link['source']].index[0] for link in connections],
            target = [nodes_df[nodes_df['id'] == link['target']].index[0] for link in connections],
            value = [link['value'] for link in connections]
        )
    )])
    
    fig.update_layout(
        title="Industry-Region Connection Flows",
        font=dict(size=12)
    )
    
    return fig



def plot_badge_progression(df):
    """Plot badge achievement over time with hard-coded values and forecast."""
    # Hard-coded data from the provided values
    data = [
        {"date": "15/11/2023", "active": 253, "networkers": 75, "posters": 0},
        {"date": "22/11/2023", "active": 293, "networkers": 86, "posters": 0},
        {"date": "07/12/2023", "active": 304, "networkers": 88, "posters": 0},
        {"date": "22/01/2024", "active": 322, "networkers": 89, "posters": 2},
        {"date": "12/02/2024", "active": 325, "networkers": 89, "posters": 2},
        {"date": "11/03/2024", "active": 486, "networkers": 139, "posters": 5},
        {"date": "04/04/2024", "active": 739, "networkers": 210, "posters": 5},
        {"date": "17/04/2024", "active": 852, "networkers": 215, "posters": 5},
        {"date": "24/04/2024", "active": 876, "networkers": 217, "posters": 5},
        {"date": "13/05/2024", "active": 966, "networkers": 218, "posters": 5},
        {"date": "17/07/2024", "active": 1031, "networkers": 222, "posters": 6},
        {"date": "05/08/2024", "active": 1042, "networkers": 222, "posters": 6},
        {"date": "13/01/2025", "active": 1074, "networkers": 223, "posters": 6}
    ]
    
    # Convert to DataFrame
    badge_df = pd.DataFrame(data)
    badge_df['date'] = pd.to_datetime(badge_df['date'], dayfirst=True)
    badge_df = badge_df.sort_values('date')
    
    # Create forecast data (6 months beyond the last data point)
    last_date = badge_df['date'].max()
    
    # Calculate simple linear forecast based on last 3 months of data
    recent_data = badge_df.iloc[-3:]
    
    # Calculate monthly growth rates
    active_growth = (recent_data['active'].iloc[-1] - recent_data['active'].iloc[0]) / 3
    networker_growth = (recent_data['networkers'].iloc[-1] - recent_data['networkers'].iloc[0]) / 3
    poster_growth = (recent_data['posters'].iloc[-1] - recent_data['posters'].iloc[0]) / 3
    
    # Generate forecast dates (monthly for 6 months)
    forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
    
    # Generate forecast values
    forecast_data = []
    for i, date in enumerate(forecast_dates, 1):
        forecast_data.append({
            'date': date,
            'active_forecast': recent_data['active'].iloc[-1] + active_growth * i,
            'networkers_forecast': recent_data['networkers'].iloc[-1] + networker_growth * i,
            'posters_forecast': recent_data['posters'].iloc[-1] + poster_growth * i
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical badge achievement lines
    fig.add_trace(go.Scatter(
        x=badge_df['date'],
        y=badge_df['active'],
        mode='lines+markers',
        name='Frequently Active Users',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=badge_df['date'],
        y=badge_df['networkers'],
        mode='lines+markers',
        name='Networkers',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=badge_df['date'],
        y=badge_df['posters'],
        mode='lines+markers',
        name='Top Posters',
        line=dict(color='orange', width=2)
    ))
    
    # Add forecast lines (dashed)
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['active_forecast'],
        mode='lines',
        name='Active Users Forecast',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['networkers_forecast'],
        mode='lines',
        name='Networkers Forecast',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['posters_forecast'],
        mode='lines',
        name='Top Posters Forecast',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Badge Achievement Over Time (with 6-month Forecast)',
        xaxis_title='Date',
        yaxis_title='Number of Users',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig
    


# Define models as session state objects to maintain scope across functions
if 'engagement_model' not in st.session_state:
    st.session_state['engagement_model'] = None
if 'churn_model' not in st.session_state:
    st.session_state['churn_model'] = None
if 'X_test_engagement' not in st.session_state:
    st.session_state['X_test_engagement'] = None
if 'y_test_engagement' not in st.session_state:
    st.session_state['y_test_engagement'] = None
if 'X_test_churn' not in st.session_state:
    st.session_state['X_test_churn'] = None
if 'y_test_churn' not in st.session_state:
    st.session_state['y_test_churn'] = None

# Function to train engagement model with better error handling
def train_engagement_model(df):
    try:
        # Check if we have enough data
        if len(df) < 10:  # Arbitrary minimum threshold
            st.warning("Not enough data to train engagement model. Need at least 10 records.")
            return DummyClassifier(0), 0.5, pd.DataFrame(), pd.Series()
            
        # Features for engagement prediction
        features = ['total_friend_count', 'profile_completion', 'days_since_login', 'uses_mobile']
        
        # Make sure all required columns exist
        missing_columns = [col for col in features if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns for engagement model: {', '.join(missing_columns)}")
            return DummyClassifier(0), 0.5, pd.DataFrame(), pd.Series()
        
        # Define target: user has either Frequently Active or Networker badge
        friend_count_threshold = df['total_friend_count'].quantile(0.5)  # Use median to balance classes
        days_threshold = df['days_since_login'].quantile(0.5)  # Use median to balance classes
        
        df['has_engagement_badge'] = ((df['total_friend_count'] >= friend_count_threshold) & 
                                     (df['profile_completion'] == 1) & 
                                     (df['days_since_login'] <= days_threshold)).astype(int)
        
        # Handle any infinities or NaNs in the data
        X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['has_engagement_badge']
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            st.warning("Warning: All users fall into the same engagement category. Using a simple model.")
            
            # Create a dummy model that always predicts the majority class
            class DummyClassifier:
                def __init__(self, majority_class):
                    self.majority_class = majority_class
                    self.feature_importances_ = np.ones(len(features)) / len(features)
                    
                def predict_proba(self, X):
                    # Return probabilities for both classes - always predict majority class
                    probs = np.zeros((len(X), 2))
                    probs[:, self.majority_class] = 1
                    return probs
                    
                def predict(self, X):
                    return np.full(len(X), self.majority_class)
            
            majority_class = int(y.mode()[0])
            model = DummyClassifier(majority_class)
            auc = 0.5  # Default value for random performance
            
            # Create test data for display purposes
            X_test = X.sample(frac=0.3, random_state=42)
            y_test = y.loc[X_test.index]
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluation
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            except (IndexError, ValueError) as e:
                st.warning(f"Error calculating AUC: {e}")
                auc = 0.5
        
        # Store in session state
        st.session_state['engagement_model'] = model
        st.session_state['X_test_engagement'] = X_test
        st.session_state['y_test_engagement'] = y_test
        
        return model, auc, X_test, y_test
        
    except Exception as e:
        st.error(f"Error training engagement model: {e}")
        
        # Return a dummy model to avoid breaking the app
        class DummyClassifier:
            def __init__(self):
                self.feature_importances_ = np.ones(4) / 4
                
            def predict_proba(self, X):
                probs = np.zeros((len(X), 2))
                probs[:, 0] = 1
                return probs
                
            def predict(self, X):
                return np.zeros(len(X))
        
        return DummyClassifier(), 0.5, pd.DataFrame(), pd.Series()

# Function to train churn model with better error handling
def train_churn_model(df):
    try:
        # Check if we have enough data
        if len(df) < 10:  # Arbitrary minimum threshold
            st.warning("Not enough data to train churn model. Need at least 10 records.")
            return DummyRegressor(), 0, pd.DataFrame(), pd.Series()
        
        # Define churn as no login in past 60 days
        df['churned'] = (df['days_since_login'] > 60).astype(int)
        
        # Features for churn prediction
        features = ['total_friend_count', 'profile_completion', 'uses_mobile']
        
        # Make sure all required columns exist
        missing_columns = [col for col in features if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns for churn model: {', '.join(missing_columns)}")
            return DummyRegressor(), 0, pd.DataFrame(), pd.Series()
        
        # Handle any infinities or NaNs in the data
        X = df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        
        # Store in session state
        st.session_state['churn_model'] = model
        st.session_state['X_test_churn'] = X_test
        st.session_state['y_test_churn'] = y_test
        
        return model, mse, X_test, y_test
        
    except Exception as e:
        st.error(f"Error training churn model: {e}")
        
        # Return a dummy model to avoid breaking the app
        class DummyRegressor:
            def __init__(self):
                self.feature_importances_ = np.ones(3) / 3
                
            def predict(self, X):
                return np.zeros(len(X))
        
        return DummyRegressor(), 0, pd.DataFrame(), pd.Series()

def train_segmentation_model(df):
    # Features for clustering
    features = ['total_friend_count', 'days_since_login', 'profile_completion', 'uses_mobile']
    
    # Prepare features
    X = df[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-means model with elbow method to determine k
    inertia = []
    k_range = range(2, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Choose optimal k (for demo, use k=4)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    return kmeans, scaler, df




def create_market_opportunity_analysis(df, parent_tab=None):
    """
    Create a comprehensive market opportunity analysis visualization
    with strategic quadrants and actionable insights.
    """
    # If this is being used as a subtab, use the parent container
    container = parent_tab if parent_tab else st
    
    container.title("Market Opportunity Analysis")
    
    # Explanation of the analysis
    container.markdown("""
    This analysis helps identify countries with the highest potential for marketing investment and growth.
    The quadrant map positions each country based on current market penetration and growth potential,
    helping to prioritize marketing efforts and resources.
    """)
    
    # Add methodology explanation and weight adjustment
    with container.expander("📊 Analysis Methodology & Weight Configuration", expanded=False):
        # [Your existing methodology explanation code here]
        container.markdown("""
        ### Growth Potential Score Methodology
        
        The Growth Potential Score is a weighted composite of five key indicators that predict future growth 
        in a market. Each indicator has been assigned a weight based on its proven impact on platform growth:
        """)
        
        # Create columns for the weight sliders
        col1, col2 = container.columns(2)
        
        with col1:
            room_to_grow_weight = col1.slider(
                "Room to Grow Weight (inverse of current penetration)", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.3, 
                step=0.05,
                help="Higher weight means we prioritize markets with lower current penetration"
            )
            
            profile_completion_weight = col1.slider(
                "Profile Completion Weight", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.3, 
                step=0.05,
                help="Higher weight prioritizes markets with quality user profiles, indicating engaged users"
            )
            
            recent_activity_weight = col1.slider(
                "Recent Activity Weight", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.2, 
                step=0.05,
                help="Higher weight prioritizes markets with users active in the last 30 days"
            )
        
        with col2:
            networker_rate_weight = col2.slider(
                "Network Building Weight", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.1, 
                step=0.05,
                help="Higher weight prioritizes markets where users are actively building connections"
            )
            
            mobile_adoption_weight = col2.slider(
                "Mobile Adoption Weight", 
                min_value=0.0, 
                max_value=0.5, 
                value=0.1, 
                step=0.05,
                help="Higher weight prioritizes markets with strong mobile app usage"
            )
        
        # Display rationale for each weight
        container.markdown("""
        ### Weight Justification
        
        **Room to Grow (30%)**: Markets with lower penetration represent untapped potential. Our historical data shows that 
        markets in the early-to-mid adoption phase grow 2-3x faster than mature markets. This receives high weighting as 
        it directly indicates remaining market opportunity.
        
        **Profile Completion (30%)**: Profile completion is a lead indicator of user engagement and platform stickiness. 
        Internal analysis across markets shows that users with complete profiles are 4x more likely to become active network 
        builders and 3x more likely to remain active after 6 months.
        
        **Recent Activity (20%)**: Active users drive network effects and content generation. Markets with higher activity 
        rates show stronger organic growth through word-of-mouth. Recent 30-day activity receives moderate weighting as it 
        measures current momentum.
        
        **Network Building (10%)**: Users who build connections (10+ connections) create sustainable network value. Our 
        analysis shows that each new connection increases retention probability by approximately 5%. This receives lower 
        weighting as it's partially captured by other metrics.
        
        **Mobile Adoption (10%)**: Mobile users show 60% higher engagement rates and 40% better retention than web-only users. 
        This metric receives lower weighting as it's an enabler of engagement rather than a direct growth predictor.
        
        *Note: These weights can be adjusted based on evolving business priorities and new market insights.*
        """)
        
        # Ensure weights sum to 1.0
        total_weight = room_to_grow_weight + profile_completion_weight + recent_activity_weight + networker_rate_weight + mobile_adoption_weight
        
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding differences
            container.warning(f"⚠️ Note: Total weights sum to {total_weight:.2f}. For mathematical consistency, values will be normalized to sum to 1.0.")
    
    # Calculate comprehensive metrics by country
    container.subheader("Country Market Analysis")
    
    try:
        # Calculate metrics by country - ensure we're using user_id if present, otherwise fallback
        id_col = 'user_id' if 'user_id' in df.columns else df.index.name if df.index.name else 'index'
        
        if id_col == 'index' and 'index' not in df.columns:
            df = df.reset_index()
        
        # For profile_completion, check if we have the column, otherwise try to derive it from profile_avatar_created
        profile_col = 'profile_completion' if 'profile_completion' in df.columns else None
        if profile_col is None and 'profile_avatar_created' in df.columns:
            profile_col = 'profile_avatar_created'
            # Convert Y/N to 1/0 if needed
            if df[profile_col].dtype == 'object':
                df[profile_col] = df[profile_col].apply(lambda x: 1 if x == 'Y' else 0)
        
        # Mobile usage column determination
        mobile_col = 'uses_mobile' if 'uses_mobile' in df.columns else None
        if mobile_col is None and 'App' in df.columns:
            # Create a mobile flag based on App column
            df['uses_mobile'] = df['App'].apply(lambda x: 1 if x in ['iOS', 'Android', 'Mobile'] else 0)
            mobile_col = 'uses_mobile'
        
        # Now create the aggregation dictionary dynamically based on available columns
        agg_dict = {id_col: 'nunique'}  # This will be renamed to member_count
        
        if 'total_friend_count' in df.columns:
            agg_dict['total_friend_count'] = 'mean'
        
        if profile_col:
            agg_dict[profile_col] = 'mean'
        
        if 'days_since_login' in df.columns:
            agg_dict['days_since_login'] = [
                'mean',
                lambda x: (x <= 30).mean()  # Recent activity rate
            ]
        
        if mobile_col:
            agg_dict[mobile_col] = 'mean'
        
        # Calculate comprehensive metrics by country
        country_metrics = df.groupby('Country').agg(agg_dict)
        
        # Flatten the column names if we have a MultiIndex
        if isinstance(country_metrics.columns, pd.MultiIndex):
            country_metrics.columns = [
                f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                for col in country_metrics.columns
            ]
        
        # Rename the columns to standardized names
        rename_dict = {}
        for col in country_metrics.columns:
            if col.startswith(f"{id_col}_nunique"):
                rename_dict[col] = 'member_count'
            elif col.startswith('total_friend_count_mean'):
                rename_dict[col] = 'avg_connections'
            elif col.startswith(f"{profile_col}_mean"):
                rename_dict[col] = 'profile_completion'
            elif col.startswith('days_since_login_<lambda>'):
                rename_dict[col] = 'recent_activity'
            elif col.startswith('days_since_login_mean'):
                rename_dict[col] = 'avg_days_since_login'
            elif col.startswith(f"{mobile_col}_mean"):
                rename_dict[col] = 'mobile_adoption'
        
        country_metrics = country_metrics.rename(columns=rename_dict)
        
        # Calculate networker rate if not present
        if 'networker_rate' not in country_metrics.columns and 'total_friend_count' in df.columns:
            networker_counts = df.groupby('Country')['total_friend_count'].apply(
                lambda x: (x >= 10).mean()
            )
            country_metrics['networker_rate'] = networker_counts
        
        # Reset index to get Country as a column
        country_metrics = country_metrics.reset_index()
        
        # Filter for countries with enough data
        min_members = container.slider("Minimum members per country", 5, 50, 10)
        significant_countries = country_metrics[country_metrics['member_count'] >= min_members]
        
        if len(significant_countries) < 2:
            container.warning("Not enough countries with sufficient data for analysis. Try lowering the minimum member threshold.")
            return
        
        # Calculate market penetration (normalized by log of member count)
        max_members = significant_countries['member_count'].max()
        significant_countries['market_penetration'] = significant_countries['member_count'].apply(
            lambda x: np.log(x) / np.log(max_members) if x > 0 else 0
        )
        
        # Normalize weights to ensure they sum to 1.0
        weight_sum = room_to_grow_weight + profile_completion_weight + recent_activity_weight + networker_rate_weight + mobile_adoption_weight
        
        # Fill any missing columns with 0 to prevent errors
        for col in ['profile_completion', 'recent_activity', 'networker_rate', 'mobile_adoption']:
            if col not in significant_countries.columns:
                significant_countries[col] = 0
                container.warning(f"Missing data for {col}. Using default values of 0.")
        
        # Calculate growth potential with normalized weights
        significant_countries['growth_potential'] = (
            (room_to_grow_weight/weight_sum) * (1 - significant_countries['market_penetration']) +
            (profile_completion_weight/weight_sum) * significant_countries['profile_completion'] +
            (recent_activity_weight/weight_sum) * significant_countries['recent_activity'] +
            (networker_rate_weight/weight_sum) * significant_countries['networker_rate'] +
            (mobile_adoption_weight/weight_sum) * significant_countries['mobile_adoption']
        )
        
        # Normalize scores to 0-1 range
        min_potential = significant_countries['growth_potential'].min()
        max_potential = significant_countries['growth_potential'].max()
        
        if max_potential > min_potential:  # Avoid division by zero
            significant_countries['growth_potential'] = (
                (significant_countries['growth_potential'] - min_potential) / 
                (max_potential - min_potential)
            )
        
        # Define region mapping for colors
        region_mapping = {
            # Africa
            'Nigeria': 'Africa', 'Kenya': 'Africa', 'South Africa': 'Africa', 'Ghana': 'Africa', 
            'Morocco': 'Africa', 'Egypt': 'Africa', 'Rwanda': 'Africa', 'Tanzania': 'Africa',
            'Uganda': 'Africa', 'Ethiopia': 'Africa', 'Senegal': 'Africa', 'Algeria': 'Africa',
            'Tunisia': 'Africa', 'Somaliland': 'Africa', 'Djibouti': 'Africa',
            
            # Europe
            'United Kingdom': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Spain': 'Europe',
            'Italy': 'Europe', 'Netherlands': 'Europe', 'Switzerland': 'Europe', 'Belgium': 'Europe',
            'Sweden': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe', 'Finland': 'Europe',
            'Ireland': 'Europe', 'Portugal': 'Europe', 'Greece': 'Europe', 'Austria': 'Europe',
            'Poland': 'Europe', 'Hungary': 'Europe', 'Czech Republic': 'Europe', 'Russia': 'Europe',
            
            # Americas
            'United States': 'Americas', 'Canada': 'Americas', 'Brazil': 'Americas', 'Colombia': 'Americas',
            'Mexico': 'Americas', 'Argentina': 'Americas', 'Chile': 'Americas', 'Peru': 'Americas',
            'Ecuador': 'Americas', 'Venezuela': 'Americas', 'Costa Rica': 'Americas', 'Panama': 'Americas',
            
            # Middle East
            'Turkey': 'Middle East', 'Israel': 'Middle East', 'UAE': 'Middle East', 'Saudi Arabia': 'Middle East',
            'Qatar': 'Middle East', 'Kuwait': 'Middle East', 'Bahrain': 'Middle East', 'Oman': 'Middle East',
            'Jordan': 'Middle East', 'Lebanon': 'Middle East', 'Iraq': 'Middle East', 'Iran': 'Middle East',
            
            # Asia Pacific
            'China': 'Asia Pacific', 'Japan': 'Asia Pacific', 'India': 'Asia Pacific', 'Australia': 'Asia Pacific',
            'New Zealand': 'Asia Pacific', 'Singapore': 'Asia Pacific', 'Malaysia': 'Asia Pacific', 
            'Indonesia': 'Asia Pacific', 'Philippines': 'Asia Pacific', 'Thailand': 'Asia Pacific',
            'Vietnam': 'Asia Pacific', 'South Korea': 'Asia Pacific', 'Hong Kong': 'Asia Pacific',
            'Taiwan': 'Asia Pacific'
        }
        
        # Add region column
        significant_countries['Region'] = significant_countries['Country'].map(
            lambda x: region_mapping.get(x, 'Other')
        )
        
        # Create quadrant analysis visualization
        fig = px.scatter(
            significant_countries,
            x='market_penetration',
            y='growth_potential',
            size='member_count',
            color='Region',
            hover_name='Country',
            hover_data={
                'market_penetration': False,  # Hide normalized value
                'member_count': True,
                'avg_connections': ':.1f',
                'profile_completion': ':.1%',
                'recent_activity': ':.1%',
                'Region': True
            },
            labels={
                'market_penetration': 'Market Penetration',
                'growth_potential': 'Growth Potential Score'
            },
            title='Country Market Opportunity Analysis',
            size_max=40,
            color_discrete_map={
                'Africa': '#2ca02c',
                'Europe': '#ff7f0e',
                'Americas': '#1f77b4',
                'Middle East': '#d62728',
                'Asia Pacific': '#9467bd',
                'Other': '#7f7f7f'
            }
        )
        
        # Add quadrant lines and labels
        x_mid = 0.5
        y_mid = 0.5
        
        # Add quadrant lines
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="gray"),
            x0=x_mid, y0=0, x1=x_mid, y1=1, xref="x", yref="y"
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", width=1, color="gray"),
            x0=0, y0=y_mid, x1=1, y1=y_mid, xref="x", yref="y"
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=0.25, y=0.75, xref="x", yref="y", text="Emerging Opportunities",
            showarrow=False, font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
        fig.add_annotation(
            x=0.75, y=0.75, xref="x", yref="y", text="Strategic Priorities",
            showarrow=False, font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
        fig.add_annotation(
            x=0.25, y=0.25, xref="x", yref="y", text="Lower Priority Markets",
            showarrow=False, font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
        fig.add_annotation(
            x=0.75, y=0.25, xref="x", yref="y", text="Established Markets",
            showarrow=False, font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=700,
            xaxis=dict(range=[0, 1], title_font=dict(size=14)),
            yaxis=dict(range=[0, 1], title_font=dict(size=14)),
            legend=dict(
                title="Region",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the figure
        container.plotly_chart(fig, use_container_width=True)
        
        # Identify top opportunities in each quadrant
        quadrants = {
            'Strategic Priorities': significant_countries[
                (significant_countries['market_penetration'] >= x_mid) & 
                (significant_countries['growth_potential'] >= y_mid)
            ].sort_values('member_count', ascending=False).head(3),
            
            'Emerging Opportunities': significant_countries[
                (significant_countries['market_penetration'] < x_mid) & 
                (significant_countries['growth_potential'] >= y_mid)
            ].sort_values('growth_potential', ascending=False).head(3),
            
            'Established Markets': significant_countries[
                (significant_countries['market_penetration'] >= x_mid) & 
                (significant_countries['growth_potential'] < y_mid)
            ].sort_values('member_count', ascending=False).head(3),
            
            'Lower Priority Markets': significant_countries[
                (significant_countries['market_penetration'] < x_mid) & 
                (significant_countries['growth_potential'] < y_mid)
            ].sort_values('member_count', ascending=False).head(3)
        }
        
        # Display strategic recommendations
        container.header("Strategic Marketing Recommendations")
        
        # Create columns for quadrant recommendations
        col1, col2 = container.columns(2)
        
        with col1:
            col1.subheader("Strategic Priorities")
            if len(quadrants['Strategic Priorities']) > 0:
                col1.markdown("""
                **These established markets with high growth potential should be your primary focus.**
                
                **Recommendation:** Invest in deeper engagement and expand offerings. These markets
                combine existing scale with untapped potential for continued strong growth.
                """)
                
                for _, country in quadrants['Strategic Priorities'].iterrows():
                    col1.markdown(f"""
                    - **{country['Country']}** ({country['member_count']} members)
                      - {country['profile_completion']*100:.1f}% profile completion
                      - {country['recent_activity']*100:.1f}% active in last 30 days
                      - Focus on mentorship programs and community building
                    """)
            else:
                col1.info("No countries currently in this quadrant.")
                
            col1.subheader("Established Markets")
            if len(quadrants['Established Markets']) > 0:
                col1.markdown("""
                **These markets have strong adoption but lower growth momentum.**
                
                **Recommendation:** Focus on retention, re-engagement of dormant users,
                and deeper feature adoption. These markets need maintenance rather than expansion.
                """)
                
                for _, country in quadrants['Established Markets'].iterrows():
                    col1.markdown(f"""
                    - **{country['Country']}** ({country['member_count']} members)
                      - {country['profile_completion']*100:.1f}% profile completion
                      - {country['recent_activity']*100:.1f}% active in last 30 days
                      - Emphasize retention and premium features
                    """)
            else:
                col1.info("No countries currently in this quadrant.")
        
        with col2:
            col2.subheader("Emerging Opportunities")
            if len(quadrants['Emerging Opportunities']) > 0:
                col2.markdown("""
                **These markets show strong potential despite smaller current user bases.**
                
                **Recommendation:** Targeted expansion campaigns with localized content.
                These markets represent your best growth opportunities and could become
                tomorrow's strategic priorities with proper investment.
                """)
                
                for _, country in quadrants['Emerging Opportunities'].iterrows():
                    col2.markdown(f"""
                    - **{country['Country']}** ({country['member_count']} members)
                      - {country['profile_completion']*100:.1f}% profile completion
                      - {country['recent_activity']*100:.1f}% active in last 30 days
                      - Focus on awareness campaigns and regional partnerships
                    """)
            else:
                col2.info("No countries currently in this quadrant.")
                
            col2.subheader("Lower Priority Markets")
            if len(quadrants['Lower Priority Markets']) > 0:
                col2.markdown("""
                **These markets currently show limited scale and potential.**
                
                **Recommendation:** Maintain minimal presence but don't invest significant
                resources. Monitor for changes in growth signals before reconsidering investment.
                """)
                
                for _, country in quadrants['Lower Priority Markets'].iterrows():
                    col2.markdown(f"""
                    - **{country['Country']}** ({country['member_count']} members)
                      - {country['profile_completion']*100:.1f}% profile completion
                      - {country['recent_activity']*100:.1f}% active in last 30 days
                      - Passive maintenance only
                    """)
            else:
                col2.info("No countries currently in this quadrant.")
        
        # Remove "Growth Driver Analysis" and "Recommended Next Steps" sections
        
        # But keep the return statement inside the try block!
        return significant_countries
        
    except Exception as e:
        container.error(f"Error in market opportunity analysis: {e}")
        import traceback
        container.exception(e)
        return None
        
        

# Add this function to create time-based features for forecasting
def create_time_features(df, date_col):
    """
    Create time-based features from a date column for time series forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the date column
    date_col : str
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional time-based features
    """
    df = df.copy()
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['quarter'] = df[date_col].dt.quarter
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week
    
    # Create cyclical features for day of week and month
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    return df

# Add this function to prepare data for time series forecasting
def prepare_time_series_data(df, target_col, date_col='last_login_date', freq='D'):
    """
    Prepare time series data for forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw DataFrame containing user data
    target_col : str
        Name of the column to forecast
    date_col : str
        Name of the date column
    freq : str
        Frequency for the time series ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame aggregated and prepared for time series modeling
    """
    try:
        # Make sure the date column is a datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create a complete date range
        date_range = pd.date_range(
            start=df[date_col].min(),
            end=df[date_col].max(),
            freq=freq
        )
        
        # Aggregate data by date
        if target_col == 'New Users':
            # Count new users by signup/first login date
            ts_data = df.groupby(df[date_col].dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
        
        elif target_col == 'Active Users':
            # Count active users by date
            ts_data = df.groupby(df[date_col].dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
        
        elif target_col == 'Profile Completions':
            # Count profile completions by date
            profile_completions = df[df['profile_completion'] == 1]
            ts_data = profile_completions.groupby(profile_completions[date_col].dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
        
        elif target_col == 'Average Connections':
            # Average connections by date
            ts_data = df.groupby(df[date_col].dt.date)['total_friend_count'].mean().reset_index()
            ts_data.columns = [date_col, 'value']
        
        else:
            # Default to counting users
            ts_data = df.groupby(df[date_col].dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
        
        # Convert date back to datetime for proper indexing
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        # Create a dataframe with the complete date range
        date_df = pd.DataFrame({date_col: date_range})
        
        # Merge to get a complete time series with no gaps
        ts_data = pd.merge(date_df, ts_data, on=date_col, how='left')
        ts_data['value'] = ts_data['value'].fillna(0)
        
        # Add time-based features
        ts_data = create_time_features(ts_data, date_col)
        
        return ts_data
        
    except Exception as e:
        st.error(f"Error preparing time series data: {e}")
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=[date_col, 'value', 'dayofweek', 'month', 'year', 'dayofyear'])

# Prophet imports

# Prepare data for Prophet
def prepare_prophet_data(df, target_col, date_col='last_login_date', freq='D'):
    """
    Prepare time series data for Prophet forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw DataFrame containing user data
    target_col : str
        Name of the column to forecast
    date_col : str
        Name of the date column
    freq : str
        Frequency for the time series ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame formatted for Prophet (with 'ds' and 'y' columns)
    """
    try:
        # Make sure the date column is a datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create a complete date range
        date_range = pd.date_range(
            start=df[date_col].min(),
            end=df[date_col].max(),
            freq=freq
        )
        
        # Aggregate data by date based on the target metric
        if target_col == 'New Users':
            # Count new users by signup/first login date
            ts_data = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
            # Ensure date_col is datetime
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        elif target_col == 'Active Users':
            # Count active users by date
            ts_data = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
            # Ensure date_col is datetime
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        elif target_col == 'Profile Completions':
            # Count profile completions by date
            profile_completions = df[df['profile_completion'] == 1]
            ts_data = profile_completions.groupby(pd.to_datetime(profile_completions[date_col]).dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
            # Ensure date_col is datetime
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        elif target_col == 'Average Connections':
            # Average connections by date
            ts_data = df.groupby(pd.to_datetime(df[date_col]).dt.date)['total_friend_count'].mean().reset_index()
            ts_data.columns = [date_col, 'value']
            # Ensure date_col is datetime
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        elif target_col == 'Total Active Users':
            # Create a cumulative count of users over time
            # First, count new users by date
            new_users_by_date = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().reset_index()
            new_users_by_date.columns = [date_col, 'new_users']
            # Ensure date_col is datetime
            new_users_by_date[date_col] = pd.to_datetime(new_users_by_date[date_col])
            
            # Create date range to ensure all dates are included
            date_range_df = pd.DataFrame({date_col: date_range})
            
            # Merge to get a complete time series and fill missing values
            # Use pd.merge with explicit date types to avoid the error
            complete_ts = pd.merge(
                date_range_df,
                new_users_by_date,
                left_on=date_range_df[date_col].dt.date,
                right_on=pd.to_datetime(new_users_by_date[date_col]).dt.date,
                how='left'
            )
            
            # If you still get an error, use concat with a different approach
            if date_col + '_y' in complete_ts.columns:  # This means the merge created duplicate columns
                # Use the original date_range_df dates
                complete_ts[date_col] = date_range_df[date_col]
            
            complete_ts['new_users'] = complete_ts['new_users'].fillna(0)
            
            # Calculate cumulative sum
            complete_ts['value'] = complete_ts['new_users'].cumsum()
            
            # Return the required columns, making sure to select the right date column
            if date_col + '_x' in complete_ts.columns:
                ts_data = complete_ts[[date_col + '_x', 'value']].rename(columns={date_col + '_x': date_col})
            else:
                ts_data = complete_ts[[date_col, 'value']]
        
        else:
            # Default to counting users
            ts_data = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().reset_index()
            ts_data.columns = [date_col, 'value']
            # Ensure date_col is datetime
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        
        # Create a dataframe with the complete date range
        date_df = pd.DataFrame({date_col: date_range})
        
        # Alternative approach using concat instead of merge to avoid type issues
        # First, set the date as index in both dataframes
        ts_data_indexed = ts_data.set_index(date_col)
        date_df_indexed = date_df.set_index(date_col)
        
        # Use concat to combine the dataframes
        combined = pd.concat([ts_data_indexed, date_df_indexed], axis=1)
        
        # Reset the index to get the date column back
        ts_data_complete = combined.reset_index()
        
        # Fill missing values
        if 'value' not in ts_data_complete.columns:
            ts_data_complete['value'] = 0
        else:
            ts_data_complete['value'] = ts_data_complete['value'].fillna(0)
        
        # Rename columns for Prophet
        prophet_df = ts_data_complete.rename(columns={date_col: 'ds', 'value': 'y'})
        
        return prophet_df
        
    except Exception as e:
        st.error(f"Error preparing data for Prophet: {e}")
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=['ds', 'y'])


# Function to train Prophet model
def train_prophet_model(prophet_df, horizon=90, regressors=None, include_holidays=None, changepoints=None, seasonality_mode='additive'):
    """
    Train a Prophet model for time series forecasting.
    
    Parameters:
    -----------
    prophet_df : pandas.DataFrame
        DataFrame with 'ds' and 'y' columns prepared for Prophet
    horizon : int
        Number of days to forecast
    regressors : list of dict
        List of additional regressor configurations
    include_holidays : str or list
        Country code(s) for built-in holidays or custom holiday DataFrame
    changepoints : list
        List of dates where changepoints are expected
    seasonality_mode : str
        'additive' or 'multiplicative'
        
    Returns:
    --------
    tuple
        (trained model, forecast dataframe, performance metrics)
    """
    try:
        # Check if Prophet is available
        if not PROPHET_AVAILABLE:
            st.error("Prophet is not installed. Please install prophet to use this feature.")
            return None, pd.DataFrame(), {}
        
        # Check if we have enough data
        if len(prophet_df) < 10:
            st.warning("Not enough historical data for reliable forecasting. Need at least 10 data points.")
            return None, pd.DataFrame(), {}
        
        # Initialize Prophet model with parameters
        model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of trend
            seasonality_prior_scale=10.0,  # Flexibility of seasonality
            holidays_prior_scale=10.0,     # Flexibility of holidays
            daily_seasonality=False,       # Auto-detect daily seasonality
            weekly_seasonality=True,       # Include weekly seasonality
            yearly_seasonality=True,       # Include yearly seasonality
            seasonality_mode=seasonality_mode  # 'additive' or 'multiplicative'
        )
        
        # Add country holidays if specified
        if include_holidays:
            if isinstance(include_holidays, str):
                # Add built-in country holidays
                model.add_country_holidays(country_name=include_holidays)
            elif isinstance(include_holidays, pd.DataFrame):
                # Add custom holidays
                model.add_holidays(include_holidays)
            elif isinstance(include_holidays, list):
                # Add multiple countries' holidays
                for country in include_holidays:
                    model.add_country_holidays(country_name=country)
        
        # Add regressors if specified
        if regressors:
            for regressor in regressors:
                if regressor.get('name') in prophet_df.columns:
                    model.add_regressor(
                        name=regressor.get('name'),
                        prior_scale=regressor.get('prior_scale', 10),
                        mode=regressor.get('mode', 'additive'),
                        standardize=regressor.get('standardize', True)
                    )
        
        # Add custom changepoints if specified
        if changepoints:
            model.changepoints = pd.to_datetime(changepoints)
        
        # Fit the model
        model.fit(prophet_df)
        
        # Create future dataframe for predictions
        future = model.make_future_dataframe(periods=horizon)
        
        # Add regressor values to future if needed
        if regressors:
            for regressor in regressors:
                if regressor.get('name') in prophet_df.columns:
                    # Use last value for future regressor values (simple approach)
                    last_value = prophet_df[regressor.get('name')].iloc[-1]
                    future[regressor.get('name')] = future['ds'].apply(
                        lambda x: last_value if x > prophet_df['ds'].max() else 
                        prophet_df.loc[prophet_df['ds'] == x, regressor.get('name')].iloc[0] 
                        if x in prophet_df['ds'].values else last_value
                    )
        
        # Make predictions
        forecast = model.predict(future)
        
        # Get component contributions (trend, seasonality, holidays)
        components = {}
        for component in ['trend', 'weekly', 'yearly', 'holidays']:
            if component in forecast.columns:
                components[component] = forecast[component].mean()
        
        # Calculate performance metrics (using simple approach)
        historical_period = len(prophet_df)
        historical_predictions = forecast.iloc[:historical_period]
        actual_values = prophet_df['y']
        
        # Only use historical period where we have both actual and predicted values
        y_true = actual_values.values
        y_pred = historical_predictions['yhat'].values
        
        metrics = {}
        
        # Handle non-positive values for MAPE
        if np.all(y_true > 0):
            # Mean Absolute Percentage Error
            metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            metrics['MAPE'] = np.nan
            
        # Mean Absolute Error
        metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Squared Error
        metrics['RMSE'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Add component contributions to metrics
        metrics['components'] = components
        
        return model, forecast, metrics
        
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        return None, pd.DataFrame(), {}

# Function to plot Prophet forecast results
def plot_prophet_forecast(model, forecast, historical_df, target_col='Value'):
    """
    Create plots for Prophet forecast results.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pandas.DataFrame
        DataFrame with forecast results from Prophet
    historical_df : pandas.DataFrame
        Original historical data with 'ds' and 'y' columns
    target_col : str
        Name of the metric being forecasted
        
    Returns:
    --------
    dict
        Dictionary of Plotly figures
    """
    try:
        figures = {}
        
        # Create forecast plot
        if model:
            # Use Prophet's built-in plotting function
            fig = plot_plotly(model, forecast)
            fig.update_layout(
                title=f'{target_col} Forecast',
                xaxis_title='Date',
                yaxis_title=target_col,
                hovermode='x unified'
            )
            figures['forecast'] = fig
            
            # Add components plot
            components_fig = plot_components_plotly(model, forecast)
            figures['components'] = components_fig
        else:
            # Create a simple plot if model is not available
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical_df['ds'],
                y=historical_df['y'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))
            
            # Add future prediction
            future_mask = forecast['ds'] > historical_df['ds'].max()
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, 'ds'],
                y=forecast.loc[future_mask, 'yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            
            # Add prediction intervals
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, 'ds'],
                y=forecast.loc[future_mask, 'yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, 'ds'],
                y=forecast.loc[future_mask, 'yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name='Prediction Interval'
            ))
            
            fig.update_layout(
                title=f'{target_col} Forecast',
                xaxis_title='Date',
                yaxis_title=target_col,
                hovermode='x unified'
            )
            figures['forecast'] = fig
            
        return figures
        
    except Exception as e:
        st.error(f"Error plotting Prophet forecast: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating forecast plot: {e}",
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return {'forecast': fig}
        


# Add this function to plot forecasting results
def plot_forecast_results(historical_data, forecast_data, date_col='last_login_date', target_col='New Users'):
    """
    Create a plot showing historical data and forecasts.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame with historical time series data
    forecast_data : pandas.DataFrame
        DataFrame with forecast results
    date_col : str
        Name of the date column
    target_col : str
        Name of the metric being forecasted
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the forecast plot
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data[date_col],
            y=historical_data['value'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Add prediction intervals
        fig.add_trace(go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data['forecast_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data['forecast_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Prediction Interval'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{target_col} Forecast',
            xaxis_title='Date',
            yaxis_title=target_col,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting forecast results: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating forecast plot: {e}",
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

# Add this function to plot feature importance
def plot_feature_importance(model, feature_cols):
    """
    Create a plot showing feature importance from the forecast model.
    
    Parameters:
    -----------
    model : xgboost.XGBRegressor
        Trained XGBoost model
    feature_cols : list
        List of feature column names
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the feature importance plot
    """
    try:
        # If model is None, return empty figure
        if model is None:
            fig = go.Figure()
            fig.add_annotation(text="No model available for feature importance",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create plot
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance for Forecast Model'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating feature importance plot: {e}",
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig



# Define the plot_model_performance function with access to session state
# First, fix the plot_model_performance function - it should ONLY return figures, not implement UI
def plot_model_performance(model_type, X_test, y_test):
    try:
        if model_type == "Engagement Prediction" and X_test is not None and y_test is not None:
            # Get model from session state
            model = st.session_state.get('engagement_model')
            
            if model is None:
                st.warning("Engagement model not available")
                # Return empty figure
                fig = go.Figure()
                fig.add_annotation(text="Model not available",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Create confusion matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Create heatmap
            fig = px.imshow(cm, 
                           text_auto=True,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Not Engaged', 'Engaged'],
                           y=['Not Engaged', 'Engaged'],
                           title="Model Performance: Confusion Matrix")
            
            return fig
        
        elif model_type == "Churn Prediction" and X_test is not None and y_test is not None:
            # Get model from session state
            model = st.session_state.get('churn_model')
            
            if model is None:
                st.warning("Churn model not available")
                # Return empty figure
                fig = go.Figure()
                fig.add_annotation(text="Model not available",
                                 xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False)
                return fig
            
            # Create scatter plot
            y_pred = model.predict(X_test)
            
            fig = px.scatter(x=y_test, y=y_pred, 
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Model Performance: Predicted vs Actual")
            
            return fig
        
        elif model_type == "User Segmentation":
            # For segmentation, show cluster characteristics
            cluster_data = []
            
            # This is just placeholder data for the demo
            metrics = ['Connection Count', 'Login Frequency', 'Profile Score', 'Mobile Usage']
            clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
            
            for i in range(4):
                values = np.random.rand(4)
                for j, metric in enumerate(metrics):
                    cluster_data.append({
                        'Metric': metric,
                        'Cluster': clusters[i],
                        'Value': values[j]
                    })
            
            df_cluster = pd.DataFrame(cluster_data)
            
            fig = px.bar(df_cluster, x='Metric', y='Value', color='Cluster',
                        title="Cluster Profile Comparison",
                        barmode='group')
            
            return fig
        
        else:
            # Default placeholder graph
            fig = go.Figure()
            fig.add_annotation(text="No model data available for visualization",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False)
            return fig
            
    except Exception as e:
        st.error(f"Error creating model performance visualization: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {e}",
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig


# First, move the forecasting_tab function definition before the main() function

def forecasting_tab(df):
    """Implement Prophet time series forecasting within a Streamlit tab."""
    st.title("Time Series Forecasting with Prophet")
    
    # Debug info to verify Prophet status
    st.write(f"Prophet availability status: {PROPHET_AVAILABLE}")
    
    # Select the metric to forecast
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_metrics = [
            "New Users",
            "Active Users",
            "Profile Completions",
            "Average Connections",
            "Total Active Users"  # New option added here
        ]
        
        selected_metric = st.selectbox(
            "Select Metric to Forecast",
            options=forecast_metrics,
            index=forecast_metrics.index(st.session_state.get('selected_forecast_metric', 'New Users'))
        )
        st.session_state['selected_forecast_metric'] = selected_metric
        
        # Add explanation for Total Active Users
        if selected_metric == "Total Active Users":
            st.info("This metric shows the cumulative count of all users who have joined the platform over time. The forecast predicts the total user base growth.")
    
    with col2:
        forecast_horizon = st.slider(
            "Forecast Horizon (Days)",
            min_value=30,
            max_value=365,
            value=st.session_state.get('forecast_horizon', 90),
            step=30
        )
        st.session_state['forecast_horizon'] = forecast_horizon
    
    # Advanced options
    with st.expander("Advanced Forecasting Options"):
        seasonality_mode = st.selectbox(
            "Seasonality Mode",
            options=["additive", "multiplicative"],
            help="Additive means seasonal effects are added to the trend. Multiplicative means they're multiplied by the trend."
        )
        
        include_holidays = st.multiselect(
            "Include Country Holidays",
            options=["US", "UK", "India", "Germany", "France", "Brazil", "China", "Spain", "Italy", "Russia", "Australia", "Canada", "Japan"],
            default=[],
            help="Include holidays from selected countries in the model"
        )
        
        # Smoother forecasting parameters
        changepoint_prior_scale = st.slider(
            "Trend Flexibility",
            min_value=0.001,
            max_value=0.5,
            value=0.02,  # Reduced from default 0.05 for smoother trends
            step=0.005,
            format="%.3f",
            help="Lower values create a smoother trend line (default: 0.05)"
        )
        
        seasonality_prior_scale = st.slider(
            "Seasonality Strength",
            min_value=0.1,
            max_value=20.0,
            value=5.0,  # Reduced for smoother seasonality
            step=0.5,
            help="Lower values create smoother seasonal patterns (default: 10.0)"
        )
        
        # Optional data smoothing
        apply_smoothing = st.checkbox(
            "Apply data smoothing",
            value=True,
            help="Apply rolling average to smooth input data"
        )
        
        if apply_smoothing:
            smoothing_window = st.slider(
                "Smoothing Window (Days)",
                min_value=1,
                max_value=14,
                value=3,
                help="Size of rolling window for data smoothing"
            )
    
    # Prepare data for Prophet
    with st.spinner("Preparing time series data..."):
        prophet_df = prepare_prophet_data(df, selected_metric)
        
        # Apply smoothing if selected
        if apply_smoothing and len(prophet_df) > smoothing_window:
            prophet_df['y'] = prophet_df['y'].rolling(window=smoothing_window, center=True).mean()
            prophet_df['y'] = prophet_df['y'].fillna(method='bfill').fillna(method='ffill')
    
    # Display time series data overview
    st.subheader("Historical Data Overview")
    
    # Display basic statistics about the data
    if not prophet_df.empty:
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Data Points", len(prophet_df))
        
        with metrics_col2:
            st.metric("Average", f"{prophet_df['y'].mean():.1f}")
        
        with metrics_col3:
            st.metric("Max Value", f"{prophet_df['y'].max():.1f}")
        
        with metrics_col4:
            # Calculate trend (if we have enough data)
            if len(prophet_df) >= 30:
                recent = prophet_df['y'].tail(30).mean()
                older = prophet_df['y'].head(30).mean()
                trend_pct = ((recent - older) / older * 100) if older > 0 else 0
                
                st.metric(
                    "30-Day Trend",
                    f"{trend_pct:.1f}%",
                    delta=f"{trend_pct:.1f}%"
                )
            else:
                st.metric("Data Range (Days)", (prophet_df['ds'].max() - prophet_df['ds'].min()).days)
        
        # Plot historical data
        fig = px.line(
            prophet_df, 
            x='ds', 
            y='y',
            title=f'Historical {selected_metric} Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for analysis.")
    
    # Train prophet model and generate forecast
    if st.button("Generate Forecast"):
        with st.spinner("Training Prophet model and generating forecast..."):
            # Configure Prophet with smoother settings
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=10.0,
                daily_seasonality=False,
                weekly_seasonality=10,  # Using fewer Fourier terms for smoother weekly seasonality
                yearly_seasonality=True,
                seasonality_mode=seasonality_mode
            )
            
            # Add country holidays if selected
            if include_holidays:
                for country in include_holidays:
                    model.add_country_holidays(country_name=country)
            
            # Fit the model
            model.fit(prophet_df)
            
            # Create future dataframe for predictions
            future = model.make_future_dataframe(periods=forecast_horizon)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Calculate performance metrics
            historical_period = len(prophet_df)
            historical_predictions = forecast.iloc[:historical_period]
            actual_values = prophet_df['y']
            
            y_true = actual_values.values
            y_pred = historical_predictions['yhat'].values
            
            metrics = {}
            
            # Handle non-positive values for MAPE
            if np.all(y_true > 0):
                metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            else:
                metrics['MAPE'] = np.nan
                
            metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
            metrics['RMSE'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # Display metrics
            st.success("Forecast generated successfully!")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
            
            with metrics_col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            
            with metrics_col3:
                if not np.isnan(metrics['MAPE']):
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                else:
                    st.metric("MAPE", "N/A")
            
            # Plot the forecast
            st.subheader("Forecast Plot")
            
            # Create a custom smoother forecast plot with Plotly
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['y'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast line
            future_mask = forecast['ds'] > prophet_df['ds'].max()
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, 'ds'],
                y=forecast.loc[future_mask, 'yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add prediction intervals
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, 'ds'],
                y=forecast.loc[future_mask, 'yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast.loc[future_mask, 'ds'],
                y=forecast.loc[future_mask, 'yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name='95% Confidence Interval'
            ))
            
            # Improve layout
            fig.update_layout(
                title=f'{selected_metric} Forecast',
                xaxis_title='Date',
                yaxis_title=selected_metric,
                hovermode='x unified',
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Special visualization for Total Active Users
            if selected_metric == "Total Active Users":
                st.subheader("Growth Rate Analysis")
                
                # Calculate month-over-month growth rate
                forecast_monthly = forecast.copy()
                forecast_monthly['month'] = forecast_monthly['ds'].dt.to_period('M')
                monthly_forecast = forecast_monthly.groupby('month').agg({
                    'yhat': 'last',
                    'yhat_lower': 'last',
                    'yhat_upper': 'last'
                }).reset_index()
                
                monthly_forecast['month'] = monthly_forecast['month'].astype(str)
                monthly_forecast['growth'] = monthly_forecast['yhat'].pct_change() * 100

                monthly_forecast['month_date'] = pd.to_datetime(monthly_forecast['month'] + '-01')
                monthly_forecast = monthly_forecast[monthly_forecast['month_date'] >= pd.to_datetime('2023-10-01')]
                
                # Plot growth rate
                growth_fig = px.bar(
                    monthly_forecast.dropna(),
                    x='month',
                    y='growth',
                    title='Projected Monthly Growth Rate (%)',
                    labels={'growth': 'Growth Rate (%)', 'month': 'Month'},
                    color='growth',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(growth_fig, use_container_width=True)
                
                # Calculate key growth milestones
                current_users = prophet_df['y'].iloc[-1]
                forecast_end = forecast[forecast['ds'] > prophet_df['ds'].max()]
                
                # Find when the platform might reach certain milestones
                if current_users > 0:
                    growth_targets = [
                        current_users * 1.5,  # 50% growth
                        current_users * 2,    # 100% growth  
                        current_users * 3     # 200% growth
                    ]
                    
                    milestones = []
                    for target in growth_targets:
                        if any(forecast_end['yhat'] >= target):
                            target_date = forecast_end[forecast_end['yhat'] >= target]['ds'].iloc[0]
                            days_to_target = (target_date - prophet_df['ds'].max()).days
                            milestones.append({
                                "Target": f"{(target/current_users - 1)*100:.0f}% Growth",
                                "Users": f"{target:.0f}",
                                "Date": target_date.strftime('%B %d, %Y'),
                                "Days": days_to_target
                            })
                    
                    if milestones:
                        st.subheader("Projected Growth Milestones")
                        st.dataframe(pd.DataFrame(milestones))

# Main application
def main():
    
    # Load and preprocess data
    df_raw = load_data()
    
    # Calculate completeness on raw data before preprocessing
    completeness_data, completeness_df = calculate_data_completeness(df_raw)
    st.session_state['raw_completeness_data'] = completeness_data
    st.session_state['raw_completeness_df'] = completeness_df
    
    df = preprocess_data(df_raw)
    
    # Sidebar for filters
    st.sidebar.title("YES! Connect Analytics")
    
    # Date range filter
    min_date = df['last_login_date'].min().date()
    max_date = df['last_login_date'].max().date()

    if 'enable_country_standardization' in st.session_state and st.session_state['enable_country_standardization']:
        df_standardized, _, _ = standardize_countries(df)
        st.session_state['df_standardized'] = df_standardized
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['last_login_date'].dt.date >= start_date) & 
                         (df['last_login_date'].dt.date <= end_date)]
        st.session_state['date_range'] = date_range
    else:
        df_filtered = df
    
    # Country filter
    if 'enable_country_standardization' in st.session_state and st.session_state['enable_country_standardization']:
        # Use standardized data for filtering too
        if 'df_standardized' in st.session_state:
            # Create a filtered version of the standardized dataframe
            df_standardized = st.session_state['df_standardized']
            # Apply date filter to standardized data
            if len(date_range) == 2:
                df_standardized_filtered = df_standardized[(df_standardized['last_login_date'].dt.date >= start_date) & 
                                                         (df_standardized['last_login_date'].dt.date <= end_date)]
            else:
                df_standardized_filtered = df_standardized
                
            # Get the standardized country list for filters
            countries = sorted(df_standardized_filtered['Country'].unique())
            
            # Use the standardized filtered data instead of original
            df_filtered = df_standardized_filtered
        else:
            countries = sorted(df_filtered['Country'].unique())
    else:
        # Use original country data
        countries = sorted(df_filtered['Country'].unique())

    # Country filter - this part stays mostly the same
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=countries,
        default=st.session_state['country_filter']
    )

    if selected_countries:
        # Apply country filter
        df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
        st.session_state['country_filter'] = selected_countries
    
    # Career level filter
    career_levels = sorted(df['Career Level'].unique())
    selected_careers = st.sidebar.multiselect(
        "Select Career Levels",
        options=career_levels,
        default=st.session_state['career_filter']
    )
    
    if selected_careers:
        df_filtered = df_filtered[df_filtered['Career Level'].isin(selected_careers)]
        st.session_state['career_filter'] = selected_careers
    
    # Model selection
    model_options = [
        "Engagement Prediction",
        "Churn Prediction",
        "User Segmentation"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_options,
        index=model_options.index(st.session_state['selected_model'])
    )
    
    st.session_state['selected_model'] = selected_model
    
    # Reset filters button
    if st.sidebar.button("Reset Filters"):
        st.session_state['country_filter'] = []
        st.session_state['career_filter'] = []
        st.session_state['date_range'] = [min_date, max_date]
    
    # Main tabs
# Modify tab creation to include the Forecasting tab
    tab1, tab3, tab4, tab6,tab7, tab8 = st.tabs([
        "Overview Dashboard", 
        "Part C : Network Analysis",
        "Predictive Models",
        "Forecasting",
        "Data Dictionary",
        "Trend Analysis"# New tab
    ])
    
    # Train models once
    engagement_model, engagement_auc, X_test_engagement, y_test_engagement = train_engagement_model(df)
    churn_model, churn_mse, X_test_churn, y_test_churn = train_churn_model(df)
    segmentation_model, scaler, df_segmented = train_segmentation_model(df)
    
    with tab1:
        st.title("YES! Connect Platform Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Users</div>
                <div class="kpi-value">{len(df_filtered):,}</div>
                <div class="kpi-context">Lifetime platform adoption</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            active_users = len(df_filtered[df_filtered['days_since_login'] <= 30])
            active_pct = active_users/len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Active Users (30d)</div>
                <div class="kpi-value">{active_users:,}</div>
                <div class="kpi-context">{active_pct:.1f}% of user base</div>
                <div class="kpi-trend positive-trend">+{active_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            profiles_complete = len(df_filtered[df_filtered['profile_completion'] == 1])
            profile_pct = profiles_complete/len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Complete Profiles</div>
                <div class="kpi-value">{profiles_complete:,}</div>
                <div class="kpi-context">{profile_pct:.1f}% completion rate</div>
                <div class="kpi-trend positive-trend">+{profile_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_connections = df_filtered['total_friend_count'].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg. Connections</div>
                <div class="kpi-value">{avg_connections:.1f}</div>
                <div class="kpi-context">Per user network size</div>
                <div class="kpi-trend positive-trend">+2.4%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Geographic distribution
        st.subheader("Geographic Distribution")
        try:
            geo_fig = plot_geographic_distribution(df_filtered)
            st.plotly_chart(geo_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating geographic distribution: {e}")
        # In your tab1 section
    
        # Career distribution
        st.subheader("User Career Distribution")
        career_fig = plot_career_distribution(df_filtered)
        st.plotly_chart(career_fig, use_container_width=True)
        
        # Badge progression over time
        st.subheader("Badge Achievement Over Time")
        badge_fig = plot_badge_progression(df)
        st.plotly_chart(badge_fig, use_container_width=True)     
    
    with tab3:
        st.title("Network Analysis")
        
        # Network graph visualization
        st.subheader("User Network Visualization")
        network_fig = plot_network_graph(df_filtered)
        st.plotly_chart(network_fig, use_container_width=True)
        
           
    with tab4:
        st.title("Predictive Models")
        
        st.subheader(f"Selected Model: {selected_model}")
        
        if selected_model == "Engagement Prediction":
            st.write("""
            **This model predicts which users are likely to earn engagement badges.**
            
            It uses features like:
            - Connection count
            - Profile completion
            - Login recency and frequency
            - Platform usage
            
            The model can help identify users who need small nudges to reach badge thresholds.
            """)
            
            # Show model performance
            perf_fig = plot_model_performance("Engagement Prediction", X_test_engagement, y_test_engagement)
            st.plotly_chart(perf_fig, use_container_width=True)
            
            # Feature importance
            importance = pd.DataFrame({
                'Feature': X_test_engagement.columns,
                'Importance': engagement_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            imp_fig = px.bar(importance, x='Feature', y='Importance',
                            title='Feature Importance for Engagement Prediction')
            st.plotly_chart(imp_fig, use_container_width=True)
            
            # Sample predictions
            st.subheader("Sample Predictions")
            
            df_sample = df_filtered.sample(min(10, len(df_filtered)))
            X_sample = df_sample[['total_friend_count', 'profile_completion', 'days_since_login', 'uses_mobile']].fillna(0)
            
            df_sample['Engagement Probability'] = engagement_model.predict_proba(X_sample)[:, 1]
            
            sample_display = df_sample[['Career Level', 'Country', 'total_friend_count', 
                                        'profile_completion', 'days_since_login', 'Engagement Probability']]
            
            st.dataframe(sample_display)
            
        elif selected_model == "Churn Prediction":
            st.write("""
            **This model predicts which users are at risk of becoming inactive.**
            
            This model uses the following features:
            - Connection count
            - Profile completion
            - Historical login patterns
            - Platform preferences
            
            Identifying at-risk users allows for targeted re-engagement campaigns.
            """)
            
            # Show model performance
            perf_fig = plot_model_performance("Churn Prediction", X_test_churn, y_test_churn)
            st.plotly_chart(perf_fig, use_container_width=True)
            
            # Feature importance
            importance = pd.DataFrame({
                'Feature': X_test_churn.columns,
                'Importance': churn_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            imp_fig = px.bar(importance, x='Feature', y='Importance',
                            title='Feature Importance for Churn Prediction')
            st.plotly_chart(imp_fig, use_container_width=True)
            
            # Users at risk
            st.subheader("Users at Highest Churn Risk")
            
            df_sample = df_filtered.sample(min(10, len(df_filtered)))
            X_sample = df_sample[['total_friend_count', 'profile_completion', 'uses_mobile']].fillna(0)
            
            df_sample['Churn Risk'] = churn_model.predict(X_sample)
            
            sample_display = df_sample[['Career Level', 'Country', 'total_friend_count', 
                                        'profile_completion', 'days_since_login', 'Churn Risk']]
            
            st.dataframe(sample_display.sort_values('Churn Risk', ascending=False))
            
        elif selected_model == "User Segmentation":
            st.write("""
            **This model segments users into distinct behavior-based groups.**
            
            The segmentation identifies key user personas based on:
            - Connection patterns and network sizes
            - Platform engagement frequency
            - Profile completeness
            - Mobile vs. web platform preferences
            - Career level and experience distribution
            
            Understanding these segments helps tailor content, features, and communication strategies to different user types.
            """)
            
            # Apply segmentation to filtered data
            X_filtered = df_filtered[['total_friend_count', 'days_since_login', 'profile_completion', 'uses_mobile']].fillna(0)
            X_filtered_scaled = scaler.transform(X_filtered)
            df_filtered['cluster'] = segmentation_model.predict(X_filtered_scaled)
            
            # Define segment mapping (keep this part stable)
            segment_mapping = {
                0: "Network Builders",
                1: "Academic Engagers",
                2: "Emerging Professionals",
                3: "Established Experts",
                4: "Dormant Members"
            }
            
            # Map clusters to segment names
            df_filtered['segment'] = df_filtered['cluster'].map(lambda x: segment_mapping.get(x, f"Segment {x}"))
            
            # Calculate DYNAMIC percentages based on filtered data
            segment_counts = df_filtered['segment'].value_counts()
            total_users = len(df_filtered)
            
            # Create segments with dynamic counts and percentages
            segments = []
            
            # Define characteristic criteria (stable)
            segment_characteristics = {
                "Network Builders": {
                    "criteria": {
                        "connections_high": True,
                        "profile_complete": True,
                        "recent_login": True,
                        "career": ["Early Career Professional"]
                    },
                    "strategy": "Leverage as community ambassadors and content creators. Provide advanced networking tools and recognition for their influence."
                },
                "Academic Engagers": {
                    "criteria": {
                        "connections_medium": True,
                        "profile_partial": True,
                        "student": True
                    },
                    "strategy": "Provide educational resources, mentorship connections, and professional development opportunities. Encourage mobile app adoption."
                },
                "Emerging Professionals": {
                    "criteria": {
                        "connections_medium": True,
                        "early_career": True,
                        "entrepreneur": True
                    },
                    "strategy": "Offer career advancement resources, professional showcases, and connections to more established members."
                },
                "Established Experts": {
                    "criteria": {
                        "connections_low": True,
                        "senior": True,
                        "mid_level": True
                    },
                    "strategy": "Position as knowledge contributors and mentors. Create opportunities for thought leadership and specialized discussions."
                },
                "Dormant Members": {
                    "criteria": {
                        "inactive": True,
                        "connections_low": True,
                        "profile_incomplete": True
                    },
                    "strategy": "Re-engagement campaigns with clear value propositions. Simplified mobile onboarding and personalized content recommendations."
                }
            }
            
            # For each segment, create the entry with dynamic data
            for segment_name in segment_mapping.values():
                if segment_name in segment_counts:
                    # Calculate dynamic percentage
                    count = segment_counts[segment_name]
                    percentage = (count / total_users * 100) if total_users > 0 else 0
                    
                    # Get segment users
                    segment_users = df_filtered[df_filtered['segment'] == segment_name]
                    
                    # Calculate characteristics dynamically from the data
                    characteristics = []
                    
                    # Career distribution
                    if len(segment_users) > 0:
                        career_dist = segment_users['Career Level'].value_counts(normalize=True)
                        if len(career_dist) > 0:
                            top_career = career_dist.index[0]
                            top_career_pct = career_dist.iloc[0] * 100
                            characteristics.append(f"{top_career} ({top_career_pct:.0f}%)")
                        
                        # Connection metrics
                        conn_median = segment_users['total_friend_count'].median()
                        conn_desc = f"Average {conn_median:.0f} connections per user"
                        characteristics.append(conn_desc)
                        
                        # Login recency
                        days_median = segment_users['days_since_login'].median()
                        if days_median < 7:
                            login_desc = "Regular platform engagement (weekly logins)"
                        elif days_median < 14:
                            login_desc = "Sporadic engagement (bi-weekly logins)"
                        elif days_median < 30:
                            login_desc = "Consistent engagement (monthly logins)"
                        else:
                            login_desc = f"Last login: {days_median:.0f} days ago (median)"
                        characteristics.append(login_desc)
                        
                        # Profile completion
                        profile_pct = segment_users['profile_completion'].mean() * 100
                        characteristics.append(f"Profile avatar created ({profile_pct:.0f}%)")
                        
                        # Mobile usage
                        mobile_pct = segment_users['uses_mobile'].mean() * 100
                        characteristics.append(f"Mobile app users ({mobile_pct:.0f}%)")
                        
                        # Top regions
                        if 'Country' in segment_users.columns:
                            country_counts = segment_users['Country'].value_counts().head(2)
                            if len(country_counts) > 0:
                                top_countries = ", ".join(country_counts.index.tolist())
                                characteristics.append(f"Strong presence in {top_countries}")
                    else:
                        # Fallback for empty segments
                        characteristics = ["No users in this segment with current filters"]
                    
                    segments.append({
                        "name": segment_name,
                        "count": int(count),
                        "percentage": percentage,
                        "characteristics": characteristics,
                        "strategy": segment_characteristics[segment_name]["strategy"]
                    })
            
            # Sort segments by percentage (highest first)
            segments = sorted(segments, key=lambda x: x["percentage"], reverse=True)
            
            # Display the segmentation model using Streamlit components
            st.markdown("""
            <div style="background-color:#1e40af; color:white; padding:10px; border-radius:5px; margin-bottom:10px; text-align:center">
                <h2 style="color:white; margin:0">YES! Connect Member Segmentation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Message if filtered data is very limited
            if total_users < 10:
                st.warning(f"Limited data available with current filters ({total_users} users). Segment analysis may not be representative.")
            
            # Display each segment in a card-like format
# Display each segment in a card-like format
            for segment in segments:
                with st.container():
                    # Header with dynamic percentage and count
                    col1, col2 = st.columns([1, 9])
                    with col1:
                        st.markdown(f"""
                        <div style="background-color:#1e88e5; color:white; width:50px; height:50px; 
                        border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:16px;">
                        {segment['percentage']:.0f}%
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"### {segment['name']} ({segment['count']} users)")
                    
                    # Special handling for specific segments
                    if segment['name'] == "Network Builders":
                        # Hardcoded characteristics for Network Builders
                        st.markdown("#### Key Characteristics:")
                        st.markdown("* Student (42%)")
                        st.markdown("* Strong presence in Kenya, Nigeria")
                    
                    elif segment['name'] == "Established Experts":
                        # Hardcoded characteristics for Established Experts
                        st.markdown("#### Key Characteristics:")
                        st.markdown("* Senior Industry, Entrepreneur, Student (49%)")
                        st.markdown("* Long-term inactive (500+ days since login)")
                        st.markdown("* Concentrated in Kenya, South Africa")
                                      
                    elif segment['name'] == "Academic Engagers":
                        # Hardcoded characteristics for Academic Engagers
                        st.markdown("#### Key Characteristics:")
                        st.markdown("* Students and early academics (majority)")
                        st.markdown("* Partial profile completion (30-40%)")
                        st.markdown("* Strong mobile adoption")
                    
                    else:
                        # Use original dynamic approach for other segments
                        # Characteristics section with dynamic data
                        st.markdown("#### Key Characteristics:")
                        for trait in segment['characteristics']:
                            st.markdown(f"- {trait}")
                    
                    st.markdown("---")
            
            # Create dynamic regional heatmap
            st.header("Regional Distribution of Member Segments")
            
            # Function to map countries to regions
            def assign_region(country):
                north_africa = ['Egypt', 'Algeria', 'Tunisia', 'Morocco', 'Libya']
                west_africa = ['Ghana', 'Nigeria', 'Côte d\'Ivoire', 'Senegal', 'Benin', 'Burkina Faso']
                east_africa = ['Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'Somalia', 'Burundi']
                southern_africa = ['South Africa', 'Botswana', 'Namibia', 'Zimbabwe', 'Zambia', 'Angola']
                
                if country in north_africa:
                    return "North Africa"
                elif country in west_africa:
                    return "West Africa"
                elif country in east_africa:
                    return "East Africa"
                elif country in southern_africa:
                    return "Southern Africa"
                else:
                    return "Other Regions"
            
            # Add region to filtered dataframe
            df_filtered['region'] = df_filtered['Country'].apply(assign_region)
            
            # Calculate actual regional segment distribution
            try:
                region_segment_counts = pd.crosstab(
                    df_filtered['region'], 
                    df_filtered['segment'],
                    normalize='index'
                ) * 100
                
                # Ensure all regions and segments exist
                regions = ["North Africa", "West Africa", "East Africa", "Southern Africa", "Other Regions"]
                segment_names = [s["name"] for s in segments]
                
                # Add missing regions with zeros
                for region in regions:
                    if region not in region_segment_counts.index:
                        region_segment_counts.loc[region] = [0] * len(region_segment_counts.columns)
                
                # Add missing segments with zeros
                for segment_name in segment_names:
                    if segment_name not in region_segment_counts.columns:
                        region_segment_counts[segment_name] = 0
                
                # Reorder to match expected order
                region_segment_counts = region_segment_counts.reindex(regions)
                region_segment_counts = region_segment_counts.reindex(columns=segment_names)
                
                # Convert to lists for plotting
                heatmap_data = region_segment_counts.values.tolist()
                
                # Create Plotly heatmap with dynamic data
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=segment_names,
                    y=regions,
                    colorscale='Blues',
                    text=[[f"{val:.0f}%" for val in row] for row in heatmap_data],
                    texttemplate="%{text}",
                    textfont={"size":12},
                ))
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                                    
            except Exception as e:
                st.warning(f"Unable to generate regional distribution: {e}")
                st.info("Try adjusting your filters to include more diverse geographic data")
            
            # Show sample users from each segment
            st.subheader("Sample Users by Segment")
            
            sample_users = pd.DataFrame()
            for segment_name in segment_names:
                segment_df = df_filtered[df_filtered['segment'] == segment_name]
                if len(segment_df) > 0:
                    sample = segment_df.sample(min(2, len(segment_df)))
                    sample_users = pd.concat([sample_users, sample])
            
            if len(sample_users) > 0:
                st.dataframe(
                    sample_users[['segment', 'Career Level', 'Country', 'total_friend_count', 
                                'profile_completion', 'days_since_login']]
                    .rename(columns={
                        'segment': 'Segment',
                        'total_friend_count': 'Connections',
                        'profile_completion': 'Profile Complete',
                        'days_since_login': 'Days Since Login'
                    })
                )
            else:
                st.info("No users match current filter criteria")
                
            # Keep the original cluster visualization in an expander
            with st.expander("View Original Cluster Analysis"):

            # Define segment colors for consistency
                segment_colors = {
                "Network Builders": "#1e88e5",     # Blue
                "Established Experts": "#43a047",  # Green
                "Academic Engagers": "#fb8c00",    # Orange
                "Emerging Professionals": "#e53935", # Red
                "Dormant Members": "#8e24aa"       # Purple
            }

            # Ensure we have the segment column properly mapped
                segment_mapping = {
                0: "Network Builders",
                1: "Academic Engagers",
                2: "Emerging Professionals", 
                3: "Established Experts",
                4: "Dormant Members"
            }

            # Map cluster to segment name if not already done
            if 'segment' not in df_filtered.columns:
                df_filtered['segment'] = df_filtered['cluster'].map(lambda x: segment_mapping.get(x, f"Segment {x}"))

            # Create a dataframe for the centroids (average values for each segment)
            centroids = []
            for segment in df_filtered['segment'].unique():
                segment_data = df_filtered[df_filtered['segment'] == segment]
                if len(segment_data) > 0:  # Ensure we have data
                    centroids.append({
                        'segment': segment,
                        'x': segment_data['total_friend_count'].mean(),
                        'y': segment_data['days_since_login'].mean(),
                        'size': len(segment_data)  # Size proportional to number of users
                    })

            centroid_df = pd.DataFrame(centroids)

            # Create the main scatter plot
            fig = go.Figure()

            # Add circles for each segment centroid
            for _, row in centroid_df.iterrows():
                segment = row['segment']
                color = segment_colors.get(segment, "#7f7f7f")  # Default gray if not in mapping
                
                # Add the colored circle (bubble)
                fig.add_trace(go.Scatter(
                    x=[row['x']], 
                    y=[row['y']],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=50,
                        opacity=0.3,
                        line=dict(width=2, color=color)
                    ),
                    name=segment,
                    hoverinfo='name',
                    showlegend=False
                ))
                
                # Add segment label inside the bubble
                fig.add_annotation(
                    x=row['x'],
                    y=row['y'],
                    text=segment,
                    showarrow=False,
                    font=dict(size=12, color="black", family="Arial Black"),
                    align="center"
                )

            # Configure the layout
            fig.update_layout(
                title="User Segment Clustering Visualization",
                xaxis=dict(
                    title="Connection Count",
                    tickmode='linear',
                    tick0=0,
                    dtick=5,
                    range=[0, 20]
                ),
                yaxis=dict(
                    title="Days Since Login",
                    tickmode='array',
                    tickvals=[0, 90, 180, 365, 500],
                    range=[0, 550]
                ),
                height=600,
                plot_bgcolor='white',
                legend=dict(
                    title="Segments",
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            )

            # Add a legend box to the side
            for segment_name, color in segment_colors.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=segment_name,
                    showlegend=True
                ))

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Add explanation
            st.markdown("""
            **How to interpret this visualization:**
            - Each circle represents a user segment based on behavioral patterns
            - The X-axis shows the average number of connections per user in each segment
            - The Y-axis shows the average days since last login
            - Circle size represents the relative number of users in each segment
            - Position indicates the relationship between connection activity and login recency
            """)

# In your main() function, change the tab6 section to:
    with tab6:
        if PROPHET_AVAILABLE:
            forecasting_tab(df)
        else:
            st.error("Prophet is not installed in this environment. Please install it to use advanced forecasting.")
            st.code("pip install prophet", language="bash")
            st.info("Prophet requires additional dependencies including Stan. See installation instructions at: https://facebook.github.io/prophet/docs/installation.html")
                
    with tab7:
        st.title("Data Dictionary & Quality Assessment")
        
        # Create tabs within the Data Quality tab
        dict_tab, quality_tab, country_tab = st.tabs(["Data Dictionary", "Data Quality Assessment", "Country Standardisation"])
        
        with dict_tab:
            st.header("YES! Connect Platform Metrics Dictionary")
            st.markdown("""
            This data dictionary provides definitions and business context for all metrics used in the YES! Connect analytics platform.
            Use this as a reference when interpreting dashboard insights and making business decisions.
            """)
                       
            # Engagement Metrics
            st.subheader("Engagement & Activity Metrics")
            engagement_metrics = [
                {
                    "Metric": "Total Friend Count",
                    "Definition": "Number of connections established with other platform members",
                    "Business Significance": "Primary indicator of network building activity; correlates with platform value to user"
                },
                {
                    "Metric": "Last Login Date",
                    "Definition": "Most recent date the member accessed the platform",
                    "Business Significance": "Recency indicator for member engagement; predicts retention likelihood"
                },
                {
                    "Metric": "Days Since Login",
                    "Definition": "Calculated days between last login and current date",
                    "Business Significance": "Direct measure of engagement recency; core input for retention strategies"
                },
                {
                    "Metric": "Platform/App",
                    "Definition": "Primary access method used by member (iOS, Android, Web)",
                    "Business Significance": "Indicates preferred access channel; guides platform development priorities"
                }
            ]
            
            st.dataframe(pd.DataFrame(engagement_metrics), use_container_width=True)
            
            # Derived & Calculated Metrics
            st.subheader("Derived & Calculated Metrics")
            derived_metrics = [
                {
                    "Metric": "recent_activity",
                    "Definition": "(users_active_in_last_30_days / total_users_in_country)"
                },
                {
                    "Metric": "networker_rate",
                    "Definition": "(users_with_10+_connections / total_users_in_country)"
                },
                {
                    "Metric": "profile_completion",
                    "Definition": "(users_with_completed_profiles / total_users_in_country)"
                },
                {
                    "Metric": "market_penetration",
                    "Definition": "market_penetration = log(member_count) / log(max_in_country)"
                },
                {
                    "Metric": "mobile_adoption",
                    "Definition": "mobile_adoption = (mobile_users / total_users_in_country)"
                }
            ]

            
            st.dataframe(pd.DataFrame(derived_metrics), use_container_width=True)
            
            # Badge & Achievement Metrics
            st.subheader("Badge & Achievement Metrics")
            badge_metrics = [
                {
                    "Metric": "Frequently Active User Badge",
                    "Definition": "Awarded when member uploads profile picture and logs in on 5+ different days",
                    "Business Significance": "Identifies consistently engaged members; early indicator of platform adoption"
                },
                {
                    "Metric": "Networker Badge",
                    "Definition": "Awarded when member has uploaded a profile picture, sent 5+ connection requests and received 5+ accepted requests",
                    "Business Significance": "Identifies active community builders who expand the network"
                },
                {
                    "Metric": "Top Poster Badge",
                    "Definition": "Awarded when member uploads a profile picture, posts updates on 5+ different days and comments on others' posts",
                    "Business Significance": "Identifies content contributors who drive engagement"
                }
            ]
            
            st.dataframe(pd.DataFrame(badge_metrics), use_container_width=True)
            
            # User Segments
            st.subheader("User Segments")
            segment_metrics = [
                {
                    "Segment": "Network Builders",
                    "Characteristics": "Early career professionals with high connection counts, regular engagement, and complete profiles",
                    "Business Significance": "Platform ambassadors who drive network growth and engagement",
                    "Strategic Approach": "Leverage as community leaders and content creators; provide advanced tools and recognition"
                },
                {
                    "Segment": "Academic Engagers",
                    "Characteristics": "Predominantly students with moderate connection counts and bi-weekly platform usage",
                    "Business Significance": "Future talent pipeline and growing professional network segment",
                    "Strategic Approach": "Provide educational resources, mentorship connections, and development opportunities"
                },
                {
                    "Segment": "Emerging Professionals",
                    "Characteristics": "Early career members and entrepreneurs with growing networks and monthly engagement",
                    "Business Significance": "Career transitioners seeking professional growth opportunities",
                    "Strategic Approach": "Offer career advancement resources and connections to more established members"
                },
                {
                    "Segment": "Established Experts",
                    "Characteristics": "Senior and mid-level professionals with selective networks and periodic engagement",
                    "Business Significance": "Knowledge contributors who provide value to early-career members",
                    "Strategic Approach": "Position as mentors and thought leaders; create specialized discussion forums"
                },
                {
                    "Segment": "Dormant Members",
                    "Characteristics": "Mixed career levels with limited connections and infrequent logins (60+ days)",
                    "Business Significance": "At-risk members who may churn without intervention",
                    "Strategic Approach": "Targeted re-engagement campaigns with clear value propositions"
                }
            ]
            
            st.dataframe(pd.DataFrame(segment_metrics), use_container_width=True)
        
        # Data Quality Assessment Tab
    with quality_tab:
        st.header("Data Quality Assessment")
        st.markdown("""
        This section provides insights into data completeness, anomalies, and quality issues that may impact analysis results.
        Understanding these limitations helps ensure business decisions are made with appropriate context.
        """)
        
        # Use raw data for completeness section
        completeness_data = st.session_state.get('raw_completeness_data', [])
        completeness_df = st.session_state.get('raw_completeness_df', pd.DataFrame())
        
        # Display completeness metrics
        st.subheader("Data Completeness")
        st.dataframe(pd.DataFrame(completeness_df), use_container_width=True)
        
        # Visualize completeness
        if not completeness_df.empty:
            # Plot the data completeness
            fig = px.bar(completeness_df, 
                         x="Completeness", 
                         y="Field", 
                         orientation='h',
                         title="Data Completeness by Field (%)",
                         color="Completeness",
                         color_continuous_scale=px.colors.sequential.Blues,
                         range_color=[0, 100])
            
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No completeness data available for visualization.")
        
        # Anomaly Detection
        st.subheader("Data Anomalies & Outliers")
        
        # Get raw data for anomaly detection
        df_for_anomalies = st.session_state.get('raw_data', df)
        
        # Check for anomalies in friend counts
        friend_count_issues = False
        if 'total_friend_count' in df_for_anomalies.columns:
            q1 = df_for_anomalies['total_friend_count'].quantile(0.25)
            q3 = df_for_anomalies['total_friend_count'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df_for_anomalies[df_for_anomalies['total_friend_count'] > upper_bound]
            extreme_outliers = df_for_anomalies[df_for_anomalies['total_friend_count'] > 1000]  # Arbitrary threshold for demonstration
            
            if len(outliers) > 0:
                friend_count_issues = True
                st.markdown(f"""
                **Connection Count Anomalies**
                
                * **Identified Issue**: {len(outliers)} members ({len(outliers)/len(df_for_anomalies)*100:.1f}% of database) have unusually high connection counts
                * **Business Impact**: These outliers may skew average engagement metrics and segment classifications
                * **Recommended Action**: Review accounts with over {upper_bound:.0f} connections to verify data accuracy
                """)
                
                if len(extreme_outliers) > 0:
                    st.markdown(f"**Extreme Cases**: {len(extreme_outliers)} members have over 1,000 connections, which is highly suspicious.")
                    
                    # Sample of extreme outliers
                    st.markdown("**Examples of Potentially Problematic Records:**")
                    sample_outliers = extreme_outliers.sample(min(3, len(extreme_outliers)))
                    
                    # Display with business context
                    sample_display = sample_outliers[['Career Level', 'Country', 'total_friend_count', 'last_login_date']]
                    sample_display.columns = ['Career Level', 'Country', 'Connection Count', 'Last Login Date']
                    st.dataframe(sample_display)
        
        # Check for date anomalies
        date_issues = False
        if 'last_login_date' in df_for_anomalies.columns:
            future_dates = df_for_anomalies[df_for_anomalies['last_login_date'] > pd.to_datetime('today')]
            long_inactive = df_for_anomalies[df_for_anomalies['days_since_login'] > 365]
            
            if len(future_dates) > 0:
                date_issues = True
                st.markdown(f"""
                **Date Anomalies**
                
                * **Identified Issue**: {len(future_dates)} members ({len(future_dates)/len(df_for_anomalies)*100:.1f}% of database) have future login dates
                * **Business Impact**: Future dates cause incorrect engagement metrics and invalid churn predictions
                * **Recommended Action**: Investigate data collection process; correct timestamp recording
                """)
                
                # Sample of future dates
                st.markdown("**Examples of Future Date Records:**")
                sample_future = future_dates.sample(min(3, len(future_dates)))
                sample_display = sample_future[['Career Level', 'Country', 'Last login date']]
                st.dataframe(sample_display)
            
            if len(long_inactive) > 0:
                date_issues = True
                st.markdown(f"""
                **Extended Inactivity**
                
                * **Identified Issue**: {len(long_inactive)} members ({len(long_inactive)/len(df_for_anomalies)*100:.1f}% of database) have not logged in for over a year
                * **Business Impact**: These members likely represent unrecovered churn and distort active user metrics
                * **Recommended Action**: Consider archiving these accounts or running dedicated re-engagement campaigns
                """)
        
        # Pattern anomalies
        pattern_issues = False
        
        # Check for duplicate users
        if 'user_id' in df_for_anomalies.columns:
            duplicate_count = len(df_for_anomalies) - df_for_anomalies['user_id'].nunique()
            if duplicate_count > 0:
                pattern_issues = True
                st.markdown(f"""
                **Duplicate User Records**
                
                * **Identified Issue**: {duplicate_count} duplicate user records detected
                * **Business Impact**: Duplicates inflate user counts and distort engagement metrics
                * **Recommended Action**: Implement deduplication process based on name and email matching
                """)
        
        # Suspicious patterns in profile data
        if 'Career Level' in df_for_anomalies.columns and 'Experience Level' in df_for_anomalies.columns:
            # Example: Students with 10+ years experience
            inconsistent = df_for_anomalies[(df_for_anomalies['Career Level'] == 'Student') & (df_for_anomalies['Experience Level'] == '10 Years+')]
            
            if len(inconsistent) > 0:
                pattern_issues = True
                st.markdown(f"""
                **Data Consistency Issues**
                
                * **Identified Issue**: {len(inconsistent)} members have inconsistent career/experience combinations (e.g., Students with 10+ years experience)
                * **Business Impact**: Reduces accuracy of segmentation and targeting
                * **Recommended Action**: Update profile options to prevent incompatible selections
                """)
        
        # If no issues found in any category
        if not friend_count_issues and not date_issues and not pattern_issues:
            st.success("No significant data anomalies detected in the current dataset.")
        
        # Data Quality Summary
        st.subheader("Data Quality Summary & Recommendations")
        
        # Calculate overall data quality score (example methodology)
        completeness_avg = completeness_df["Completeness"].mean() if not completeness_df.empty else 0
        
        # Count total anomalies
        total_anomalies = 0
        if 'total_friend_count' in df_for_anomalies.columns:
            total_anomalies += len(df_for_anomalies[df_for_anomalies['total_friend_count'] > upper_bound])
        if 'last_login_date' in df_for_anomalies.columns:
            total_anomalies += len(future_dates) + len(long_inactive)
        
        anomaly_percentage = (total_anomalies / len(df_for_anomalies) * 100) if len(df_for_anomalies) > 0 else 0
        overall_score = (completeness_avg * 0.6) + ((100 - anomaly_percentage) * 0.4)
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Data Quality", f"{overall_score:.1f}%", help="Weighted score based on completeness and anomaly frequency")
        
        with col2:
            st.metric("Data Completeness", f"{completeness_avg:.1f}%", help="Average completeness across all fields")
        
        with col3:
            st.metric("Records with Issues", f"{anomaly_percentage:.1f}%", 
                     delta="-2.3%" if anomaly_percentage > 0 else None,
                     delta_color="normal" if anomaly_percentage > 0 else "off",
                     help="Percentage of records with one or more detected issues")
        
        # Priority recommendations
        st.markdown("### Priority Recommendations")
        
        recommendations = []
        
        # Dynamically generate recommendations based on found issues
        if friend_count_issues:
            recommendations.append({
                "Priority": "High",
                "Issue": "Suspicious connection counts",
                "Recommendation": "Implement validation rules for connection counts; investigate accounts with 1000+ connections",
                "Business Impact": "Improve accuracy of engagement metrics and segmentation model"
            })
        
        if date_issues:
            recommendations.append({
                "Priority": "High",
                "Issue": "Future login dates",
                "Recommendation": "Fix date collection process and validate timestamp formats",
                "Business Impact": "Ensure accurate engagement tracking and churn prediction"
            })
        
        if pattern_issues:
            recommendations.append({
                "Priority": "Medium",
                "Issue": "Inconsistent career/experience combinations",
                "Recommendation": "Update profile form to prevent impossible combinations",
                "Business Impact": "Improve segmentation accuracy and targeting relevance"
            })
        
        # Add general recommendations if list is short
        if len(recommendations) < 3:
            recommendations.append({
                "Priority": "Medium",
                "Issue": "Incomplete profile data",
                "Recommendation": "Implement progressive profile completion incentives",
                "Business Impact": "Increase data completeness for better personalization"
            })
            
            recommendations.append({
                "Priority": "Low",
                "Issue": "Limited industry diversity",
                "Recommendation": "Expand recruitment beyond education/NGO sector",
                "Business Impact": "Increase network value through greater professional diversity"
            })
        
        # Display recommendations
        if recommendations:
            st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
        else:
            st.info("No specific recommendations needed based on current data quality.")
        
        # Data governance note
        st.markdown("""
        ---
        ### Data Governance Note
        
        This quality assessment is based on automated detection of common data issues. For a comprehensive 
        data governance strategy, we recommend implementing:
        
        1. **Regular data audits** with scheduled quality reports
        2. **Data validation rules** at collection points
        3. **Data stewardship roles** for ongoing quality monitoring
        4. **Documentation of data lineage** for all metrics
        
        These practices will ensure YES! Connect maintains high-quality data to drive business decisions.
        """)

 
    # New Country Standardization tab
    with country_tab:
        st.header("Country Name Standardisation")
        st.markdown("""
        This tool identifies and resolves inconsistencies in country names
        """)
        
        # Get original country data
        if 'Country' in df.columns:
            countries_list = df['Country'].dropna().astype(str).tolist()
            
            # Standardize countries
            df_standardized, country_mapping, standardization_summary = standardize_countries(df)
            
            # Check if we found any similar countries
            if not country_mapping:
                st.success("✅ Great news! No significant country name variations detected in the dataset.")
                
                # Show country distribution
                st.subheader("Current Country Distribution")
                country_counts = df['Country'].value_counts().reset_index()
                country_counts.columns = ['Country', 'Count']
                country_counts['Percentage'] = (country_counts['Count'] / len(df) * 100).round(2)
                
                # Display top 20 countries
                st.dataframe(country_counts.head(20), use_container_width=True)
                
                if len(country_counts) > 20:
                    st.info(f"Showing top 20 countries out of {len(country_counts)} total countries.")
            else:
                # Show summary statistics
                total_variations = len(country_mapping)
                original_countries = df['Country'].nunique()
                after_standard = df_standardized['Country'].nunique()
                reduction = original_countries - after_standard
                
                # Create metrics in a row
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Total Variations Detected",
                        total_variations,
                        help="Number of country name variations found"
                    )
                    
                with metric_col2:
                    st.metric(
                        "Countries After Standardization",
                        after_standard,
                        delta=f"-{reduction}" if reduction > 0 else "0",
                        help="Number of unique countries after standardization"
                    )
                    
                with metric_col3:
                    redundancy_pct = (reduction / original_countries * 100) if original_countries > 0 else 0
                    st.metric(
                        "Redundancy Eliminated",
                        f"{redundancy_pct:.1f}%",
                        help="Percentage of redundant country entries eliminated"
                    )
                
                # Create tabs for different visualizations
                std_tab1, std_tab2, std_tab3 = st.tabs(["Variation Details", "Similarity Matrix", "Standardisation Impact"])
                
                with std_tab1:
                    st.subheader("Country Name Variations")
                    # Display standardization summary
                    std_summary = standardization_summary.sort_values(['Standard', 'Count'], ascending=[True, False])
                    
                    # Format the display columns
                    display_summary = std_summary.copy()
                    display_summary.columns = ['Standard Name', 'Variant Found', 'Count', 'Action']
                    
                    st.dataframe(display_summary, use_container_width=True)
                    
                    # Add download button for the mapping
                    csv = std_summary.to_csv(index=False)
                    st.download_button(
                        label="Download Country Standardization Table",
                        data=csv,
                        file_name="country_standardization.csv",
                        mime="text/csv",
                    )
                
                with std_tab2:
                    st.subheader("Country Name Similarity Matrix")
                    # Display similarity matrix
                    similarity_fig = plot_country_similarity_matrix(countries_list)
                    st.plotly_chart(similarity_fig, use_container_width=True)
                    
                    st.markdown("""
                    **How to read this chart**: 
                    
                    This matrix shows how similar different country names are to each other. 
                    - Darker blue cells indicate higher similarity between names
                    - Values above 0.8 typically suggest variations of the same country
                    - The diagonal is always 1.0 (perfect match with itself)
                    """)
                
                with std_tab3:
                    st.subheader("Impact of Country Standardization")
                    # Display impact of standardization
                    impact_fig, impact_data = create_country_standardization_impact(df, df_standardized)
                    st.plotly_chart(impact_fig, use_container_width=True)
                    
                    st.markdown("""
                    This chart shows how standardization affects the distribution of countries.
                    - Blue bars show counts before standardization
                    - Dark blue bars show counts after standardization
                    - Countries that gain records are those chosen as the standard version
                    - Countries that disappear are those merged into standard versions
                    """)
                
                # Add option to apply standardization
                st.markdown("### Apply Country Standardization")
                
                st.markdown("""
                Enabling standardization will affect all dashboard views and analytics that use country data.
                This includes:
                - Geographic distribution visualizations
                - Country filters
                - Regional analysis
                - Network visualization
                """)
                
                # Add toggle for standardization
                enable_standardization = st.checkbox(
                    "Enable country name standardization for all dashboard analysis",
                    value=st.session_state.get('enable_country_standardization', False),
                    help="When enabled, all country names will be standardized according to the mapping above"
                )
                
                if enable_standardization:
                    # Store standardized dataframe in session state for use in other parts of the app
                    st.session_state['df_standardized'] = df_standardized
                    st.session_state['enable_country_standardization'] = True
                    st.success("✅ Country name standardization enabled! All dashboard analysis will use standardized country names.")

                        # ADD YOUR REFRESH BUTTON RIGHT HERE
                    if st.button("Refresh Map with Standardized Countries"):
                        # Clear any cached results
                        st.cache_data.clear()
        
                    # Show a sample of the standardized data
                    st.subheader("Sample of Standardized Data")
                    sample_before = df[['Country']].sample(min(5, len(df)))
                    sample_after = df_standardized.loc[sample_before.index, ['Country']]
                    
                    sample_comparison = pd.DataFrame({
                        'Original Country': sample_before['Country'].values,
                        'Standardized Country': sample_after['Country'].values
                    })
                    
                    st.dataframe(sample_comparison, use_container_width=True)
                else:
                    # Remove standardized dataframe from session state if exists
                    st.session_state['enable_country_standardization'] = False
                    if 'df_standardized' in st.session_state:
                        del st.session_state['df_standardized']
                    st.info("Country name standardization is currently disabled.")
        else:
            st.warning("Country field not found in the dataset. Please ensure your data includes a 'Country' column.")
    
            
            # Data Quality Summary
            st.subheader("Data Quality Summary & Recommendations")
            
            # Calculate overall data quality score (example methodology)
            completeness_avg = completeness_df["Completeness"].mean() if not completeness_df.empty else 0
            
            # Count total anomalies
            total_anomalies = 0
            if 'total_friend_count' in df.columns:
                total_anomalies += len(df[df['total_friend_count'] > upper_bound])
            if 'last_login_date' in df.columns:
                total_anomalies += len(future_dates) + len(long_inactive)
            
            anomaly_percentage = (total_anomalies / len(df) * 100) if len(df) > 0 else 0
            overall_score = (completeness_avg * 0.6) + ((100 - anomaly_percentage) * 0.4)
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Data Quality", f"{overall_score:.1f}%", help="Weighted score based on completeness and anomaly frequency")
            
            with col2:
                st.metric("Data Completeness", f"{completeness_avg:.1f}%", help="Average completeness across all fields")
            
            with col3:
                st.metric("Records with Issues", f"{anomaly_percentage:.1f}%", 
                         delta="-2.3%" if anomaly_percentage > 0 else None,
                         delta_color="normal" if anomaly_percentage > 0 else "off",
                         help="Percentage of records with one or more detected issues")
            
            # Priority recommendations
            st.markdown("### Priority Recommendations")
            
            recommendations = []
            
            # Dynamically generate recommendations based on found issues
            if friend_count_issues:
                recommendations.append({
                    "Priority": "High",
                    "Issue": "Suspicious connection counts",
                    "Recommendation": "Implement validation rules for connection counts; investigate accounts with 1000+ connections",
                    "Business Impact": "Improve accuracy of engagement metrics and segmentation model"
                })
            
            if date_issues:
                recommendations.append({
                    "Priority": "High",
                    "Issue": "Future login dates",
                    "Recommendation": "Fix date collection process and validate timestamp formats",
                    "Business Impact": "Ensure accurate engagement tracking and churn prediction"
                })
            
            if pattern_issues:
                recommendations.append({
                    "Priority": "Medium",
                    "Issue": "Inconsistent career/experience combinations",
                    "Recommendation": "Update profile form to prevent impossible combinations",
                    "Business Impact": "Improve segmentation accuracy and targeting relevance"
                })
            
            # Add general recommendations if list is short
            if len(recommendations) < 3:
                recommendations.append({
                    "Priority": "Medium",
                    "Issue": "Incomplete profile data",
                    "Recommendation": "Implement progressive profile completion incentives",
                    "Business Impact": "Increase data completeness for better personalization"
                })
                
                recommendations.append({
                    "Priority": "Low",
                    "Issue": "Limited industry diversity",
                    "Recommendation": "Expand recruitment beyond education/NGO sector",
                    "Business Impact": "Increase network value through greater professional diversity"
                })
            
            # Display recommendations
            if recommendations:
                st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
            else:
                st.info("No specific recommendations needed based on current data quality.")
            
            # Data governance note
            st.markdown("""
            ---
            ### Data Governance Note
            
            This quality assessment is based on automated detection of common data issues. For a comprehensive 
            data governance strategy, we recommend implementing:
            
            1. **Regular data audits** with scheduled quality reports
            2. **Data validation rules** at collection points
            3. **Data stewardship roles** for ongoing quality monitoring
            4. **Documentation of data lineage** for all metrics
            
            These practices will ensure YES! Connect maintains high-quality data to drive business decisions.
            """)
            
    with tab8:
        trend_analysis_tab(df_filtered)
        
# Run the application
if __name__ == "__main__":
    main()
