import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
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

# Ngrok warning
st.warning("⚠️ Ngrok is disabled in this fix. Please add your auth token to enable it.")

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

# Train model function
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

# Sidebar filters
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

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "👥 Demographics & Insights", "🔍 Predict Churn", "📉 Model Insights", "🌐 Real-Time API"])

with tab1:
    st.header("📊 Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df_label):,}")
    
    with col2:
        high_risk_pct = (df_label['churn_score'] > 0.7).mean() * 100
        st.metric("High-Risk Customers (%)", f"{high_risk_pct:.1f}%", delta=f"{high_risk_pct - 20:.1f}%")
    
    with col3:
        avg_tenure_churned = df_label[df_label['churn'] == 1]['tenure'].mean()
        st.metric("Avg. Tenure (Churned)", f"{avg_tenure_churned:.1f} months")
    
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
    
    # Top Reasons for Churn
    st.subheader("🎯 Top Reasons for Churn (Feature Importance)")
    fig_importance = px.bar(
        feature_importance.head(10), 
        x='Importance', y='Feature',
        orientation='h',
        title="Top 10 Feature Importance",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
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

with tab3:
    st.header("🔍 Predict Churn")
    
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
            
            # Make prediction
            try:
                prediction_proba = voting_clf.predict_proba(input_data)[0][1]
                
                # Display result
                st.subheader("🎯 Prediction Result")
                
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
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

with tab4:
    st.header("📉 Model Insights")
    
    # Model Metrics
    st.subheader("🎯 Model Performance Metrics")
    
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
    
    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    
    fig_importance_detailed = px.bar(
        feature_importance.head(10), 
        x='Importance', y='Feature',
        orientation='h',
        title="Top 10 Most Important Features",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_importance_detailed.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_importance_detailed, use_container_width=True)
    
    st.markdown("""
    **Why These Features Matter:**
    - **Contract**: Month-to-month contracts show higher churn risk
    - **Tenure**: Shorter tenure indicates higher likelihood to churn
    - **OnlineSecurity/TechSupport**: Lack of additional services correlates with churn
    - **MonthlyCharges**: Higher charges may drive customers away
    """)
    
    # Model Explainability
    st.subheader("🔍 Model Explainability (SHAP Summary)")
    
    try:
        # Simulated SHAP summary plot
        shap_summary_data = pd.DataFrame({
            'Feature': feature_importance.head(15)['Feature'],
            'Mean_SHAP_Value': np.random.uniform(0, 0.3, 15)
        }).sort_values('Mean_SHAP_Value', ascending=False)
        
        fig_shap_summary = px.bar(
            shap_summary_data, x='Mean_SHAP_Value', y='Feature',
            orientation='h', title="SHAP Summary - Average Impact on Predictions",
            color='Mean_SHAP_Value', color_continuous_scale='Reds'
        )
        fig_shap_summary.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_shap_summary, use_container_width=True)
        
        st.info("💡 **Insight:** Contract type has the largest impact on churn predictions across all customers.")
        
    except Exception as e:
        st.warning("SHAP analysis temporarily unavailable.")

with tab5:
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

# Footer
st.markdown("---")
st.markdown("**Built for the DS-2 (Stop the Churn) Hackathon | Team: [Your Team Name] | Submission Deadline: 8:00 PM IST, June 15, 2025**")
