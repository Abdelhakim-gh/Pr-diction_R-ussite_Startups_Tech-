import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained models
best_rf = joblib.load('best_rf.pkl')
best_xgb = joblib.load('best_xgb.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Function to preprocess new data
def preprocess_data(df):
    # Feature Engineering
    df['funding_efficiency'] = df['total_funding'] / df['funding_rounds']
    df['team_efficiency'] = df['revenue_growth'] / df['team_size']
    df['tech_stack_density'] = df['tech_stack_size'] / df['team_size']
    df['patent_efficiency'] = df['patents'] / df['total_funding']
    df['social_media_impact'] = df['social_media_score'] / (df['competitors'] + 1)
    
    # Drop columns based on previous analysis
    features_to_drop = ['pivot_count', 'patent_efficiency', 'regulatory_score', 'client_retention']
    df.drop(columns=features_to_drop, axis=1, inplace=True)
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN values with the median of each column
    df.fillna(df.median(), inplace=True)
    
    # Normalize and standardize the data
    df_scaled = scaler.transform(df)
    
    return df_scaled

# Streamlit app
st.title('Startup Success Prediction')

# Upload new data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data.head())
    
    # Preprocess the new data
    new_data_preprocessed = preprocess_data(new_data)
    
    # Predictions with Random Forest
    rf_preds = best_rf.predict(new_data_preprocessed)
    rf_probs = best_rf.predict_proba(new_data_preprocessed)
    
    # Predictions with XGBoost
    xgb_preds = best_xgb.predict(new_data_preprocessed)
    xgb_probs = best_xgb.predict_proba(new_data_preprocessed)
    
    # Display predictions
    st.write("Random Forest Predictions:")
    st.write(rf_preds)
    st.write("Random Forest Probabilities:")
    st.write(rf_probs)
    
    st.write("XGBoost Predictions:")
    st.write(xgb_preds)
    st.write("XGBoost Probabilities:")
    st.write(xgb_probs)
    
    # Feature Importances
    st.write("Random Forest Feature Importances:")
    rf_feature_importances = pd.Series(best_rf.feature_importances_, index=new_data.columns)
    st.bar_chart(rf_feature_importances.sort_values(ascending=False))
    
    st.write("XGBoost Feature Importances:")
    xgb_feature_importances = pd.Series(best_xgb.feature_importances_, index=new_data.columns)
    st.bar_chart(xgb_feature_importances.sort_values(ascending=False))
    
    # Compare predictions
    comparison_df = pd.DataFrame({
        'Random Forest': rf_preds,
        'XGBoost': xgb_preds
    })
    st.write("Comparison of Predictions:")
    st.write(comparison_df)