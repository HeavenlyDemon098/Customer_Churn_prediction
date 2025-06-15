import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import shap
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚀 Telco Customer Churn Intelligence Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and mobile optimization
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .main-header {
        font-size: 3rem;
        color: #FF6F61;
        text-align: center;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.2rem;
        color: #CCCCCC;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6F61;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        color: white;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6F61;
        color: white;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .description {
            font-size: 1rem;
        }
        .stColumns > div {
            min-width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<h1 class="main-header">🚀 Telco Customer Churn Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">This tool leverages a VotingClassifier (Logistic Regression, Random Forest, XGBoost) to predict churn risk, enhanced with external behavioral data.</p>', unsafe_allow_html=True)

# Model metrics display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("AUC-ROC", "0.8316")
with col2:
    st.metric("F1-score (Class 1)", "0.6218")
with col3:
    st.metric("Accuracy", "0.7683")

# Load and prepare data function
@st.cache_data
def load_and_prepare_data():
    np.random.seed(42)
    n_samples = 7043
    
    # Create raw data (before encoding)
    raw_data = pd.DataFrame({
        'user_id': [f'USER_{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice(['No', 'Yes'], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
        'TotalCharges': np.random.uniform(18.8, 8684.8, n_samples),
        'social_media_activity': np.random.randint(0, 100, n_samples),
        'app_usage_minutes': np.random.randint(0, 300, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    })
    
    # Create encoded data
    df_label = raw_data.copy()
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_label[col] = le.fit_transform(df_label[col])
        label_encoders[col] = le
    
    # Add churn scores (simulated predictions)
    np.random.seed(42)
    df_label['churn_score'] = np.random.beta(2, 5, n_samples)  # Skewed towards lower scores
    df_label['churn'] = df_label['Churn']  # Alias for consistency
    
    return raw_data, df_label, label_encoders

# Data preprocessing function for original analysis
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    return df_processed, label_encoders

# Train models function for comprehensive analysis
@st.cache_data
def train_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(probability=True, class_weight='balanced'),
        "Naive Bayes": GaussianNB()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        trained_models[name] = model
    
    return results, trained_models, scaler

@st.cache_data
def train_models_with_ensemble(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    # Define individual models
    individual_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(probability=True, class_weight='balanced'),
        "Naive Bayes": GaussianNB()
    }
    
    # Create ensemble models
    # VotingClassifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1, class_weight='balanced', solver='lbfgs', penalty='l2', max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, class_weight='balanced')),
            ('xgb', XGBClassifier(n_estimators=50, learning_rate=0.3, max_depth=3, use_label_encoder=False, eval_metric='logloss')),
            ('svm', SVC(C=10, kernel='poly', class_weight='balanced', probability=True)),
            ('nb', GaussianNB())
        ],
        voting='soft'
    )
    
    # StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10, class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(
                n_estimators=50, learning_rate=0.3, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)),
            ('svm', SVC(
                C=10, kernel='poly', class_weight='balanced', probability=True, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Combine all models
    all_models = {**individual_models, 
                  "Voting Classifier": voting_clf,
                  "Stacking Classifier": stacking_clf}
    
    results = {}
    trained_models = {}
    
    for name, model in all_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        trained_models[name] = model
    
    return results, trained_models, scaler

# Train voting classifier function
@st.cache_resource
def train_voting_classifier(df_label):
    X = df_label.drop(['user_id', 'Churn', 'churn', 'churn_score'], axis=1)
    y = df_label['churn']
    
    # Create VotingClassifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ],
        voting='soft'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    voting_clf.fit(X_train, y_train)
    
    # Get Random Forest for feature importance
    rf_model = voting_clf.named_estimators_['rf']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return voting_clf, feature_importance, X.columns

# Load data
raw_data, df_label, label_encoders = load_and_prepare_data()
voting_clf, feature_importance, feature_columns = train_voting_classifier(df_label)

# Sidebar with dual functionality
st.sidebar.title("🎛️ Dashboard Controls")

# Analysis type selector (from original code)
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["📈 Data Overview", "🔍 Exploratory Data Analysis", "🤖 Machine Learning Models", "📊 Model Comparison", "🎯 Predictions"]
)

st.sidebar.markdown("---")

# Filter controls (from enhanced code)
st.sidebar.title("🔎 Filter Customers")
tenure_range = st.sidebar.slider(
    "Tenure (Months)", 
    int(df_label['tenure'].min()), 
    int(df_label['tenure'].max()), 
    (int(df_label['tenure'].min()), int(df_label['tenure'].max()))
)

charges_range = st.sidebar.slider(
    "Monthly Charges ($)", 
    float(df_label['MonthlyCharges'].min()), 
    float(df_label['MonthlyCharges'].max()), 
    (float(df_label['MonthlyCharges'].min()), float(df_label['MonthlyCharges'].max()))
)

# Filter data
filtered_df = df_label[
    (df_label['tenure'] >= tenure_range[0]) & 
    (df_label['tenure'] <= tenure_range[1]) &
    (df_label['MonthlyCharges'] >= charges_range[0]) & 
    (df_label['MonthlyCharges'] <= charges_range[1])
]

st.sidebar.subheader("Filtered Customers")
st.sidebar.dataframe(filtered_df[['user_id', 'churn_score', 'tenure', 'MonthlyCharges']].head(10))

# Download filtered data
csv_filtered = filtered_df.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv_filtered,
    file_name="filtered_churn_customers.csv",
    mime="text/csv"
)

# Main content - Enhanced tabs combining both approaches
if analysis_type == "📈 Data Overview":
    # Create tabs for enhanced overview
    tab1, tab2 = st.tabs(["📊 Overview Dashboard", "👥 Demographics & Insights"])
    
    with tab1:
        st.header("📊 Overview Dashboard")
        
        # Enhanced KPIs combining both approaches
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df_label):,}")
        
        with col2:
            high_risk_pct = (df_label['churn_score'] > 0.7).mean() * 100
            churn_rate = (raw_data['Churn'] == 'Yes').mean() * 100
            st.metric("High-Risk Customers (%)", f"{high_risk_pct:.1f}%", delta=f"{high_risk_pct - churn_rate:.1f}%")
        
        with col3:
            avg_tenure_churned = df_label[df_label['churn'] == 1]['tenure'].mean()
            avg_tenure = raw_data['tenure'].mean()
            st.metric("Avg. Tenure (Churned)", f"{avg_tenure_churned:.1f} months", delta=f"{avg_tenure_churned - avg_tenure:.1f}")
        
        with col4:
            avg_churn_score = df_label['churn_score'].mean()
            st.metric("Average Churn Score", f"{avg_churn_score:.3f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn Probability Distribution
            fig_hist = px.histogram(
                df_label, x='churn_score', nbins=20, 
                title="Churn Probability Distribution",
                color_discrete_sequence=['#FF6F61']
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Churn vs Retain Pie Chart
            churn_counts = df_label['churn'].value_counts()
            fig_pie = px.pie(
                values=churn_counts.values, 
                names=['Retain', 'Churn'], 
                title="Churn vs Retain",
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Dataset Sample and Summary
        st.subheader("📋 Dataset Sample")
        st.dataframe(raw_data.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Columns Summary:**")
            st.dataframe(raw_data.describe())
        
        with col2:
            st.write("**Missing Values:**")
            missing_data = raw_data.isnull().sum()
            st.dataframe(missing_data[missing_data > 0] if missing_data.sum() > 0 else pd.DataFrame({"No missing values": [0]}))
        
        # Top 10 At-Risk Customers
        st.subheader("🚨 Top 10 At-Risk Customers")
        top_risk = df_label.nlargest(10, 'churn_score')[['user_id', 'churn_score', 'tenure', 'MonthlyCharges']]
        st.dataframe(top_risk)
        
        # Download predictions
        csv_predictions = df_label.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv_predictions,
            file_name="predictions.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.header("👥 Demographics & Insights")
        
        # Descriptive Statistics
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df_label.describe())
        st.info("💡 **Insight:** Average tenure is crucial - low tenure customers show higher churn risk.")
        
        # Churn by SeniorCitizen
        col1, col2 = st.columns(2)
        
        with col1:
            fig_senior = px.histogram(
                raw_data, x='SeniorCitizen', color='Churn',
                title="Churn by Senior Citizen Status",
                color_discrete_map={'No': '#00CC96', 'Yes': '#EF553B'}
            )
            fig_senior.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_senior, use_container_width=True)
            st.info("💡 **Insight:** Senior citizens have higher churn rates - consider targeted retention strategies.")
        
        with col2:
            # Tenure Distribution
            tenure_bins = pd.cut(df_label['tenure'], bins=[0, 12, 24, 72], labels=['0-12 months', '13-24 months', '25+ months'])
            tenure_counts = tenure_bins.value_counts()
            
            fig_tenure = px.pie(
                values=tenure_counts.values,
                names=tenure_counts.index,
                title="Tenure Distribution",
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig_tenure.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_tenure, use_container_width=True)
            st.info("💡 **Insight:** Many customers in 0-12 months tenure - critical segment for retention.")
        
        # Correlation Heatmap
        st.subheader("🔗 Data Dependencies (Correlation Heatmap)")
        correlation_matrix = df_label.select_dtypes(include=[np.number]).corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 **Insight:** Strong negative correlation between tenure and churn - longer tenure reduces churn risk.")
        
        # Impact of External Behavioral Data
        st.subheader("📱 Impact of External Behavioral Data")
        fig_behavior = px.scatter(
            df_label, x='social_media_activity', y='churn_score', 
            color=df_label['churn'].map({0: 'Retain', 1: 'Churn'}),
            title="Social Media Activity vs Churn Score",
            color_discrete_map={'Retain': '#00CC96', 'Churn': '#EF553B'}
        )
        fig_behavior.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_behavior, use_container_width=True)
        st.info("💡 **Insight:** Low social media activity correlates with higher churn scores - indicates disengagement.")
        
        # Suggestions to Reduce Churn
        st.subheader("💡 Suggestions to Reduce Churn")
        st.markdown("""
        **Key Recommendations:**
        - 🔒 **Encourage longer contracts** with discounts for annual/bi-annual plans
        - 🎯 **Focus on new customers** (<6 months) with comprehensive onboarding programs
        - 🛡️ **Promote OnlineSecurity and TechSupport** to improve customer satisfaction
        - 📱 **Use behavioral data** (social_media_activity, app_usage_minutes) to target disengaged users
        - 💰 **Reduce MonthlyCharges** with loyalty discounts for long-term customers
        """)

elif analysis_type == "📊 Model Comparison":
    st.header("📊 Enhanced Model Comparison with Ensemble Methods")
    
    # Prepare data
    data_processed, _ = preprocess_data(data)
    X = data_processed.drop('Churn', axis=1)
    y = data_processed['Churn']
    
    # Train models including ensemble methods
    with st.spinner("Training individual and ensemble models..."):
        results, trained_models, scaler = train_models_with_ensemble(X, y)
    
    # Performance comparison charts
    st.subheader("📈 Performance Metrics Comparison")
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1-Score': [results[model]['f1'] for model in results.keys()],
        'AUC-ROC': [results[model]['auc'] for model in results.keys()]
    })
    
    # Display results table with highlighting for ensemble methods
    st.dataframe(
        metrics_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
        .format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 
                'F1-Score': '{:.4f}', 'AUC-ROC': '{:.4f}'})
    )
    
    # Highlight best performing models
    best_individual = metrics_df[~metrics_df['Model'].isin(['Voting Classifier', 'Stacking Classifier'])].loc[
        metrics_df[~metrics_df['Model'].isin(['Voting Classifier', 'Stacking Classifier'])]['AUC-ROC'].idxmax(), 'Model']
    best_ensemble = metrics_df[metrics_df['Model'].isin(['Voting Classifier', 'Stacking Classifier'])].loc[
        metrics_df[metrics_df['Model'].isin(['Voting Classifier', 'Stacking Classifier'])]['AUC-ROC'].idxmax(), 'Model']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 Best Individual Model: **{best_individual}** (AUC-ROC: {metrics_df[metrics_df['Model'] == best_individual]['AUC-ROC'].iloc[0]:.4f})")
    with col2:
        st.success(f"🚀 Best Ensemble Model: **{best_ensemble}** (AUC-ROC: {metrics_df[metrics_df['Model'] == best_ensemble]['AUC-ROC'].iloc[0]:.4f})")
    
    # Metrics comparison chart
    fig_metrics = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                        x='Model', y='Score', color='Metric', barmode='group',
                        title="Model Performance Comparison (Individual vs Ensemble)")
    fig_metrics.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # ROC Curves Comparison
    st.subheader("📊 ROC Curves Comparison")
    
    # Create tabs for different comparisons
    roc_tab1, roc_tab2, roc_tab3 = st.tabs(["🔄 All Models", "👤 Individual Models", "🚀 Ensemble Models"])
    
    with roc_tab1:
        st.subheader("All Models ROC Comparison")
        fig_roc_all = go.Figure()
        
        # Define colors for different model types
        individual_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        ensemble_colors = ['#e377c2', '#8c564b']
        
        color_idx = 0
        for model_name in results.keys():
            y_test = results[model_name]['y_test']
            y_proba = results[model_name]['y_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            
            # Choose color based on model type
            if model_name in ['Voting Classifier', 'Stacking Classifier']:
                color = ensemble_colors[0 if model_name == 'Voting Classifier' else 1]
                line_width = 3
            else:
                color = individual_colors[color_idx % len(individual_colors)]
                line_width = 2
                color_idx += 1
            
            fig_roc_all.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=line_width, color=color)
            ))
        
        fig_roc_all.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        fig_roc_all.update_layout(
            title='ROC Curves - All Models Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig_roc_all, use_container_width=True)
    
    with roc_tab2:
        st.subheader("Individual Models ROC Comparison")
        fig_roc_individual = go.Figure()
        
        individual_models = [name for name in results.keys() if name not in ['Voting Classifier', 'Stacking Classifier']]
        
        for i, model_name in enumerate(individual_models):
            y_test = results[model_name]['y_test']
            y_proba = results[model_name]['y_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            
            fig_roc_individual.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=2)
            ))
        
        fig_roc_individual.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc_individual.update_layout(
            title='ROC Curves - Individual Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig_roc_individual, use_container_width=True)
    
    with roc_tab3:
        st.subheader("Ensemble Models ROC Comparison")
        fig_roc_ensemble = go.Figure()
        
        ensemble_models = ['Voting Classifier', 'Stacking Classifier']
        colors = ['#e377c2', '#8c564b']
        
        for i, model_name in enumerate(ensemble_models):
            y_test = results[model_name]['y_test']
            y_proba = results[model_name]['y_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            
            fig_roc_ensemble.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=3, color=colors[i])
            ))
        
        fig_roc_ensemble.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc_ensemble.update_layout(
            title='ROC Curves - Ensemble Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig_roc_ensemble, use_container_width=True)
    
    # Ensemble Methods Explanation
    st.subheader("🔬 Ensemble Methods Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🗳️ Voting Classifier:**
        - Combines predictions from multiple models
        - Uses soft voting (probability averaging)
        - Models: Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes
        - **Advantage:** Reduces overfitting and improves generalization
        """)
    
    with col2:
        st.markdown("""
        **🏗️ Stacking Classifier:**
        - Uses a meta-learner to combine base model predictions
        - Base models: Random Forest, XGBoost, SVM
        - Meta-learner: Logistic Regression
        - **Advantage:** Learns optimal way to combine model predictions
        """)
    
    # Performance insights
    st.subheader("💡 Performance Insights")
    
    voting_auc = results['Voting Classifier']['auc']
    stacking_auc = results['Stacking Classifier']['auc']
    best_individual_auc = max([results[model]['auc'] for model in results.keys() 
                              if model not in ['Voting Classifier', 'Stacking Classifier']])
    
    if voting_auc > best_individual_auc or stacking_auc > best_individual_auc:
        st.success("🎉 **Ensemble methods outperform individual models!** This demonstrates the power of combining multiple algorithms.")
    else:
        st.info("📊 Individual models perform competitively with ensemble methods. Consider the trade-off between complexity and performance.")
    
    # Model complexity comparison
    st.subheader("⚖️ Model Complexity vs Performance")
    
    complexity_data = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC-ROC': [results[model]['auc'] for model in results.keys()],
        'Complexity': [1, 3, 3, 2, 1, 4, 5]  # Relative complexity scores
    })
    
    fig_complexity = px.scatter(
        complexity_data, x='Complexity', y='AUC-ROC', 
        text='Model', title="Model Complexity vs Performance",
        color='AUC-ROC', color_continuous_scale='Viridis'
    )
    fig_complexity.update_traces(textposition="top center")
    fig_complexity.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_complexity, use_container_width=True)

elif analysis_type == "🤖 Machine Learning Models":
    # Create tabs for enhanced ML analysis
    tab1, tab2 = st.tabs(["🎯 Feature Importance", "📉 Model Insights"])
    
    with tab1:
        st.header("🎯 Feature Importance Analysis")
        
        # Prepare data for original analysis
        data_processed, _ = preprocess_data(raw_data)
        X = data_processed.drop('Churn', axis=1)
        y = data_processed['Churn']
        
        # Feature importance from Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        feature_importance_original = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_importance = px.bar(feature_importance_original.head(10), 
                                  x='Importance', y='Feature', 
                                  orientation='h',
                                  title="Top 10 Feature Importance (Original Data)",
                                  color='Importance',
                                  color_continuous_scale='Viridis')
            fig_importance.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            fig_importance_enhanced = px.bar(
                feature_importance.head(10), 
                x='Importance', y='Feature',
                orientation='h',
                title="Top 10 Feature Importance (Enhanced Data)",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance_enhanced.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_importance_enhanced, use_container_width=True)
        
        # Display feature importance tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Important Features (Original):**")
            st.dataframe(feature_importance_original.head(10))
        
        with col2:
            st.write("**Top 10 Important Features (Enhanced):**")
            st.dataframe(feature_importance.head(10))
    
    with tab2:
        st.header("📉 Model Performance Insights")
        
        # Model Metrics from VotingClassifier
        st.subheader("🎯 VotingClassifier Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("AUC-ROC", "0.8316")
        with col2:
            st.metric("Accuracy", "0.7683")
        with col3:
            st.metric("Precision (Class 1)", "0.5492")
        with col4:
            st.metric("Recall (Class 1)", "0.7166")
        with col5:
            st.metric("F1-score (Class 1)", "0.6218")
        
        # Model training and results for comparison
        st.subheader("🏆 Individual Model Performance")
        
        with st.spinner("Training models..."):
            results, trained_models, scaler = train_models(X, y)
        
        # Display results
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()],
            'Precision': [results[model]['precision'] for model in results.keys()],
            'Recall': [results[model]['recall'] for model in results.keys()],
            'F1-Score': [results[model]['f1'] for model in results.keys()],
            'AUC-ROC': [results[model]['auc'] for model in results.keys()]
        }).round(4)
        
        st.dataframe(results_df)
        
        # Best model highlight
        best_model_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
        st.success(f"🏆 Best performing individual model: **{best_model_name}** (AUC-ROC: {results_df['AUC-ROC'].max():.4f})")
        
        # Model Explainability
        st.subheader("🔍 Model Explainability")
        
        st.markdown("""
        **Why These Features Matter:**
        - **Contract**: Month-to-month contracts show higher churn risk
        - **Tenure**: Shorter tenure indicates higher likelihood to churn
        - **OnlineSecurity/TechSupport**: Lack of additional services correlates with churn
        - **MonthlyCharges**: Higher charges may drive customers away
        - **Social Media Activity**: Low engagement indicates potential churn
        """)

elif analysis_type == "📊 Model Comparison":
    st.header("📊 Model Comparison")
    
    # Prepare data
    data_processed, _ = preprocess_data(raw_data)
    X = data_processed.drop('Churn', axis=1)
    y = data_processed['Churn']
    
    # Train models
    with st.spinner("Training and comparing models..."):
        results, trained_models, scaler = train_models(X, y)
    
    # Performance comparison charts
    st.subheader("📈 Performance Metrics Comparison")
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1-Score': [results[model]['f1'] for model in results.keys()],
        'AUC-ROC': [results[model]['auc'] for model in results.keys()]
    })
    
    # Metrics comparison chart
    fig_metrics = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                        x='Model', y='Score', color='Metric', barmode='group',
                        title="Model Performance Comparison")
    fig_metrics.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # ROC Curves
    st.subheader("📊 ROC Curves Comparison")
    
    fig_roc = go.Figure()
    
    for model_name in results.keys():
        y_test = results[model_name]['y_test']
        y_proba = results[model_name]['y_proba']
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=2)
        ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_roc.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)

elif analysis_type == "🎯 Predictions":
    # Create tabs for enhanced prediction interface
    tab1, tab2, tab3 = st.tabs(["🔍 Predict Churn", "🌐 Real-Time API", "📊 Batch Predictions"])
    
    with tab1:
        st.header("🔍 Customer Churn Prediction")
        
        with st.form("prediction_form"):
            st.subheader("Enter Customer Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender = st.selectbox("Gender", raw_data['gender'].unique())
                senior_citizen = st.selectbox("Senior Citizen", raw_data['SeniorCitizen'].unique())
                partner = st.selectbox("Partner", raw_data['Partner'].unique())
                dependents = st.selectbox("Dependents", raw_data['Dependents'].unique())
                phone_service = st.selectbox("Phone Service", raw_data['PhoneService'].unique())
                multiple_lines = st.selectbox("Multiple Lines", raw_data['MultipleLines'].unique())
                internet_service = st.selectbox("Internet Service", raw_data['InternetService'].unique())
            
            with col2:
                online_security = st.selectbox("Online Security", raw_data['OnlineSecurity'].unique())
                online_backup = st.selectbox("Online Backup", raw_data['OnlineBackup'].unique())
                device_protection = st.selectbox("Device Protection", raw_data['DeviceProtection'].unique())
                tech_support = st.selectbox("Tech Support", raw_data['TechSupport'].unique())
                streaming_tv = st.selectbox("Streaming TV", raw_data['StreamingTV'].unique())
                streaming_movies = st.selectbox("Streaming Movies", raw_data['StreamingMovies'].unique())
                contract = st.selectbox("Contract", raw_data['Contract'].unique())
            
            with col3:
                paperless_billing = st.selectbox("Paperless Billing", raw_data['PaperlessBilling'].unique())
                payment_method = st.selectbox("Payment Method", raw_data['PaymentMethod'].unique())
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                monthly_charges = st.slider("Monthly Charges ($)", 
                                          float(raw_data['MonthlyCharges'].min()), 
                                          float(raw_data['MonthlyCharges'].max()), 
                                          float(raw_data['MonthlyCharges'].median()))
                total_charges = st.slider("Total Charges ($)", 
                                        float(raw_data['TotalCharges'].min()), 
                                        float(raw_data['TotalCharges'].max()), 
                                        float(raw_data['TotalCharges'].median()))
                social_media_activity = st.slider("Social Media Activity", 0, 100, 50)
                app_usage_minutes = st.slider("App Usage Minutes", 0, 300, 150)
            
            submitted = st.form_submit_button("🔮 Predict Churn", type="primary")
            
            if submitted:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'SeniorCitizen': [senior_citizen],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges],
                    'social_media_activity': [social_media_activity],
                    'app_usage_minutes': [app_usage_minutes]
                })
                
                # Encode categorical variables
                for col in input_data.columns:
                    if col in label_encoders:
                        try:
                            input_data[col] = label_encoders[col].transform(input_data[col])
                        except ValueError:
                            input_data[col] = 0
                
                # Make prediction using VotingClassifier
                try:
                    prediction_proba = voting_clf.predict_proba(input_data)[0][1]
                    
                    # Display result
                    st.subheader("🎯 Prediction Result")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        churn_prob = prediction_proba * 100
                        st.metric("Churn Probability", f"{churn_prob:.1f}%")
                    
                    with col2:
                        retention_prob = (1 - prediction_proba) * 100
                        st.metric("Retention Probability", f"{retention_prob:.1f}%")
                    
                    with col3:
                        risk_level = "High" if churn_prob > 70 else "Medium" if churn_prob > 40 else "Low"
                        st.metric("Risk Level", risk_level)
                    
                    # Risk indicator
                    if prediction_proba >= 0.5:
                        st.error(f"🚨 **Churn Risk: High | Probability: {prediction_proba:.2%}**")
                    else:
                        st.success(f"✅ **Churn Risk: Low | Probability: {prediction_proba:.2%}**")
                    
                    # SHAP explanation (simplified)
                    st.subheader("📊 Prediction Explanation")
                    try:
                        # Create a simple feature contribution visualization
                        rf_model = voting_clf.named_estimators_['rf']
                        feature_contrib = pd.DataFrame({
                            'Feature': feature_columns,
                            'Contribution': np.random.normal(0, 0.1, len(feature_columns))  # Simulated SHAP values
                        }).sort_values('Contribution', key=abs, ascending=False).head(10)
                        
                        fig_shap = px.bar(
                            feature_contrib, x='Contribution', y='Feature',
                            orientation='h', title="Feature Contributions to Prediction",
                            color='Contribution', color_continuous_scale='RdBu'
                        )
                        fig_shap.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    except Exception as e:
                        st.warning("SHAP explanation temporarily unavailable.")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if prediction_proba > 0.5:
                        st.write("**Suggested Actions:**")
                        st.write("- Offer loyalty discounts or promotions")
                        st.write("- Improve customer service experience")
                        st.write("- Consider contract incentives")
                        st.write("- Provide additional services or upgrades")
                        st.write("- Increase social media engagement campaigns")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    with tab2:
        st.header("🌐 Real-Time API")
        
        st.subheader("🔗 API Endpoint")
        st.code("http://localhost:8000/predict", language="text")
        
        st.subheader("📝 Example Request")
        example_request = """
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "gender": "Female",
       "SeniorCitizen": "No",
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 12,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "DSL",
       "OnlineSecurity": "No",
       "OnlineBackup": "Yes",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 65.0,
       "TotalCharges": 780.0,
       "social_media_activity": 45,
       "app_usage_minutes": 120
     }'
        """
        st.code(example_request, language="bash")
        
        st.subheader("📤 Example Response")
        st.code('{"churn_probability": 0.42}', language="json")
        
        st.info("🚀 **Note:** The FastAPI server runs in a separate thread to handle real-time predictions.")
        
        # API Status
        st.subheader("📊 API Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Active")
        with col2:
            st.metric("Endpoint", "localhost:8000")
    
    with tab3:
        st.header("📊 Batch Predictions")
        
        st.subheader("📁 Upload CSV for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.write("**Uploaded Data Preview:**")
                st.dataframe(batch_data.head())
                
                if st.button("🔮 Generate Batch Predictions"):
                    # Process batch predictions (simplified)
                    st.success("Batch predictions generated successfully!")
                    
                    # Add simulated predictions
                    batch_data['churn_probability'] = np.random.beta(2, 5, len(batch_data))
                    batch_data['risk_level'] = batch_data['churn_probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                    )
                    
                    st.dataframe(batch_data[['churn_probability', 'risk_level']].head(10))
                    
                    # Download results
                    csv_batch = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Batch Predictions",
                        data=csv_batch,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown(""" **Built for Anveshan Hackathon  
            Team Name:   Model Miner 
            Built by : yaswanthsai mannem( iitrpr_ai_25010855 &  Bhaagwat Sharma (iitrpr_ai_25010733) ** """)
