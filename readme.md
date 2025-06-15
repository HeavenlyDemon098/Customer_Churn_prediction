# 🚀 Telco Customer Churn Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive machine learning dashboard for predicting and analyzing customer churn in the telecommunications industry. This tool leverages ensemble methods (VotingClassifier and StackingClassifier) to predict churn risk, enhanced with external behavioral data analysis.

![Dashboard Preview](https://via.placeholder.com/800x400/0e1117/FF6F61?text=Telco+Churn+Dashboard)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Data Schema](#-data-schema)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

### 🎯 Core Functionality
- **Interactive Dashboard**: Dark-themed, mobile-responsive Streamlit interface
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes
- **Ensemble Methods**: VotingClassifier and StackingClassifier for improved accuracy
- **Real-time Predictions**: Individual customer churn risk assessment
- **Batch Processing**: Upload CSV files for bulk predictions
- **Model Explainability**: SHAP values for prediction interpretation

### 📊 Analytics & Visualization
- **Comprehensive EDA**: Interactive exploratory data analysis
- **ROC Curve Comparisons**: Individual vs ensemble model performance
- **Feature Importance**: Identify key churn drivers
- **Customer Segmentation**: Filter and analyze customer groups
- **Behavioral Analysis**: Social media activity and app usage insights

### 🌐 API Integration
- **FastAPI Backend**: RESTful API for real-time predictions
- **JSON Response Format**: Easy integration with external systems
- **Scalable Architecture**: Designed for production deployment

## 🎮 Demo Link 



### Live Dashboard




## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
1. **Clone the repository**

2. **Create virtual environment** (recommended)

3. **Install dependencies**

4. **Run the dashboard**


## 📖 Usage

### Dashboard Navigation

#### 📈 Data Overview
- **KPI Metrics**: Total customers, churn rate, average tenure
- **Dataset Preview**: Sample data and summary statistics
- **Top At-Risk Customers**: Identify customers with highest churn probability

#### 🔍 Exploratory Data Analysis
- **Churn Distribution**: Visual breakdown of customer retention
- **Feature Analysis**: Interactive categorical and numerical feature exploration
- **Correlation Matrix**: Identify relationships between variables

#### 🤖 Machine Learning Models
- **Feature Importance**: Understand key churn drivers
- **Model Performance**: Compare individual model metrics
- **Best Model Selection**: Automatic identification of top performers

#### 📊 Model Comparison
- **Individual Models**: Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes
- **Ensemble Methods**: VotingClassifier and StackingClassifier
- **ROC Curves**: Visual performance comparison across all models
- **Performance Insights**: Automated analysis and recommendations

#### 🎯 Predictions
- **Individual Prediction**: Manual input form for single customer assessment
- **Batch Predictions**: CSV upload for multiple customer analysis
- **Real-time API**: Integration endpoint for external systems
- **SHAP Explanations**: Understand prediction reasoning

### Sidebar Controls
- **Customer Filters**: Filter by tenure and monthly charges
- **Data Export**: Download filtered datasets and predictions
- **Analysis Type Selection**: Navigate between different dashboard sections




## 📈 Model Performance

### Current Metrics (VotingClassifier)
- **AUC-ROC**: 0.8316
- **Accuracy**: 0.7683
- **F1-Score (Class 1)**: 0.6218
- **Precision (Class 1)**: 0.5492
- **Recall (Class 1)**: 0.7166

### Model Comparison Results
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7521 | 0.5234 | 0.6891 | 0.5952 | 0.8012 |
| Random Forest | 0.7634 | 0.5387 | 0.7023 | 0.6098 | 0.8145 |
| XGBoost | 0.7598 | 0.5341 | 0.6987 | 0.6054 | 0.8123 |
| SVM | 0.7456 | 0.5156 | 0.6834 | 0.5876 | 0.7934 |
| Naive Bayes | 0.7289 | 0.4987 | 0.6712 | 0.5723 | 0.7823 |
| **Voting Classifier** | **0.7683** | **0.5492** | **0.7166** | **0.6218** | **0.8316** |
| **Stacking Classifier** | **0.7701** | **0.5523** | **0.7198** | **0.6251** | **0.8334** |

### Feature Importance (Top 10)
1. **Contract** (0.156) - Contract type significantly impacts churn
2. **tenure** (0.142) - Customer tenure is crucial for retention
3. **MonthlyCharges** (0.128) - Higher charges correlate with churn
4. **TotalCharges** (0.119) - Total spending affects loyalty
5. **OnlineSecurity** (0.098) - Additional services reduce churn
6. **TechSupport** (0.087) - Support services improve retention
7. **PaymentMethod** (0.076) - Payment method influences churn
8. **InternetService** (0.071) - Service type affects satisfaction
9. **social_media_activity** (0.065) - Engagement indicates loyalty
10. **PaperlessBilling** (0.058) - Billing preference impacts churn


