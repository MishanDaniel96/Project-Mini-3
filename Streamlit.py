import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Model Training (Cached) ---
@st.cache_resource
def load_data_and_train_models():
    """Loads data, performs preprocessing, and trains both models."""
    try:
        # Assuming the CSV is in the same directory
        df = pd.read_csv("Employee-Attrition - Employee-Attrition.csv")
    except FileNotFoundError:
        st.error("Error: 'Employee-Attrition - Employee-Attrition.csv' not found. Please ensure it's in the same directory.")
        st.stop()
        
    df_raw = df.copy()

    # Preprocessing
    df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True, errors='ignore')
    df2 = df.copy()
    le = LabelEncoder()
    df2['Attrition'] = le.fit_transform(df2['Attrition']) # No->0, Yes->1

    x = df2.drop('Attrition', axis=1)
    y = df2['Attrition']
    x_encoded = pd.get_dummies(x, drop_first=True)
    all_feature_cols = x_encoded.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded, y, test_size=0.25, random_state=42, stratify=y)

    # Train Logistic Regression Model (for interactive prediction)
    logreg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(
            C=0.01,
            solver='liblinear',
            random_state=42,
            class_weight='balanced'
        ))
    ])
    logreg_model.fit(x_train, y_train)

    # Train Random Forest Model (for metrics and feature importance)
    attrition_model = RandomForestClassifier(n_estimators=100, random_state=42)
    attrition_model.fit(x_train, y_train)

    return attrition_model, logreg_model, x_test, y_test, df_raw, x_encoded.columns


# --- 2. Dashboard Functions ---

def display_metrics(model, x_test, y_test):
    # (Function body remains the same as your original script for space efficiency)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    st.header("🎯 Predictive Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AUC-ROC Score", f"{auc_roc:.4f}")
    with col2:
        st.metric("Recall (Leavers)", f"{report['1']['recall']:.2f}")
    with col3:
        st.metric("Precision (Leavers)", f"{report['1']['precision']:.2f}")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Stayed (0)', 'Actual Left (1)'], columns=['Predicted Stayed (0)', 'Predicted Left (1)'])
    st.dataframe(cm_df)

def display_top_risks(model, x_test, y_test):
    # (Function body remains the same as your original script for space efficiency)
    st.header("🚩 Top 5 Employees Flagged as High Risk")
    y_proba = model.predict_proba(x_test)[:, 1]
    results = x_test.copy()
    results['Attrition_Probability'] = y_proba
    results['Actual_Attrition'] = y_test
    at_risk_list = results.sort_values(by='Attrition_Probability', ascending=False).head(5)
    
    display_cols = ['Attrition_Probability', 'Age', 'MonthlyIncome', 'YearsAtCompany', 'Actual_Attrition']
    st.dataframe(
        at_risk_list[display_cols].style.format({"Attrition_Probability": "{:.2%}"}),
        column_config={
            "Attrition_Probability": st.column_config.ProgressColumn("Attrition Probability", format="%.2f", min_value=0, max_value=1),
            "Actual_Attrition": st.column_config.TextColumn("Actual Left (1=Yes)")
        }
    )

def make_interactive_prediction(model, all_feature_cols):
    # (Function body remains the same as the corrected version in the previous response)
    st.sidebar.header("👤 New Employee Risk Prediction")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Key Drivers Input")
    age = st.sidebar.slider('Age', 18, 60, 30)
    monthly_income = st.sidebar.number_input('Monthly Income', 1000, 20000, 5000)
    years_at_company = st.sidebar.slider('Years At Company', 0, 40, 5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Work-Life & Status")
    overtime_yes = st.sidebar.selectbox('OverTime', df_raw['OverTime'].unique())
    marital_status = st.sidebar.selectbox('Marital Status', df_raw['MaritalStatus'].unique())
    department = st.sidebar.selectbox('Department', df_raw['Department'].unique())
    
    if st.sidebar.button('Predict Attrition'):
        input_data = {col: [0] for col in all_feature_cols}
        input_df = pd.DataFrame(input_data)
        
        input_df['Age'] = age
        input_df['MonthlyIncome'] = monthly_income
        input_df['YearsAtCompany'] = years_at_company
        
        if overtime_yes == 'Yes' and 'OverTime_Yes' in input_df.columns:
            input_df['OverTime_Yes'] = 1

        if marital_status != 'Married' and f'MaritalStatus_{marital_status}' in input_df.columns:
            input_df[f'MaritalStatus_{marital_status}'] = 1

        if department != 'Research & Development' and f'Department_{department}' in input_df.columns:
            input_df[f'Department_{department}'] = 1

        # Use the correct model (LogReg pipeline)
        prediction_proba = model.predict_proba(input_df)[0, 1]
        
        st.sidebar.write("---")
        st.sidebar.subheader("Prediction Result")
        st.sidebar.write(f"### Predicted Probability of Attrition: {prediction_proba * 100:.2f}%")
        
        if prediction_proba > 0.5:
            st.sidebar.error("🚨 HIGH RISK: Intervention is recommended.")
        elif prediction_proba > 0.3:
            st.sidebar.warning("⚠️ MEDIUM RISK: Monitor closely.")
        else:
            st.sidebar.success("✅ LOW RISK: Employee is likely to stay.")

# --- NEW FUNCTION FOR EDA PLOTS ---
def display_eda(df_raw):
    st.header("📊 Exploratory Data Analysis")
    
    # Numerical Distributions
    numerical_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    axes1 = axes1.flatten()
    for i, col in enumerate(numerical_cols):
        sns.histplot(df_raw[col], kde=True, ax=axes1[i], color='#1f77b4')
        axes1[i].set_title(f'Distribution of {col}')
        axes1[i].set_xlabel(col)
        axes1[i].set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Job Role vs Gender Bar Plot
    gender_jobrole_counts = df_raw.groupby(['JobRole', 'Gender']).size().reset_index(name='Count')
    fig2, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=gender_jobrole_counts,
        x='JobRole',
        y='Count',
        hue='Gender',
        palette={'Male': 'Blue', 'Female': 'Orange'},
        ax=ax
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Employee Count by Job Role and Gender')
    plt.xlabel('Job Role')
    plt.ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig2)

# --- NEW FUNCTION FOR FEATURE IMPORTANCE ---
def display_feature_importance(model, feature_names):
    st.header("💡 Feature Importance (Model Explainability)")
    st.markdown("Top factors driving the attrition prediction.")

    # Get feature importances from Random Forest
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance_df,
        color='skyblue',
        ax=ax
    )
    ax.set_title('Top 10 Feature Importances')
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)


# --- 3. Streamlit App Layout ---

# Load data and models once
attrition_model, logreg_model, x_test, y_test, df_raw, feature_names = load_data_and_train_models()

st.set_page_config(layout="wide", page_title="Employee Attrition Dashboard")

st.title("💼 Employee Attrition Prediction Dashboard")
st.markdown("Optimized model for identifying employees at risk of turnover.")
st.markdown("---")

# Run the sidebar prediction function (LogReg model is used for robustness)
make_interactive_prediction(logreg_model, feature_names)

# Create two tabs for the main content
tab1, tab2 = st.tabs(["Prediction & Performance", "Exploratory Data Analysis (EDA)"])

with tab1:
    col_metrics, col_risks = st.columns([1, 1])

    with col_metrics:
        # Use Random Forest for main dashboard metrics
        display_metrics(attrition_model, x_test, y_test)

    with col_risks:
        # Use Random Forest for top risks
        display_top_risks(attrition_model, x_test, y_test)
    
    st.markdown("---")
    # Add Feature Importance Plot
    display_feature_importance(attrition_model, feature_names)

with tab2:
    # Add EDA plots
    display_eda(df_raw)