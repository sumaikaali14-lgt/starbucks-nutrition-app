import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Starbucks Nutrition Predictor", page_icon="☕", layout="centered")

st.title("☕ Starbucks Nutrition Predictor")
st.write("Apni pasand ke ingredients, cup size, aur milk select karein aur Calories, Protein, aur Fat check karein.")

# Model Load Karein
@st.cache_resource
def load_model():
    return joblib.load('starbucks_model.pkl')

data = load_model()
model = data['model']
feature_cols = data['feature_cols']
categorical_cols = data['categorical_cols']
numeric_cols = data['numeric_cols']
target_cols = data['target_cols']
unique_values = data['unique_values']

st.sidebar.header("Customize Your Drink")

# User Inputs Collect Karna
user_inputs = {}

for col in feature_cols:
    display_name = col.replace('_', ' ').title()
    if col in categorical_cols:
        options = [str(val) for val in unique_values[col] if pd.notna(val)]
        user_inputs[col] = st.sidebar.selectbox(f"Select {display_name}", options)
    else:
        user_inputs[col] = st.sidebar.number_input(f"Enter {display_name}", min_value=0.0, value=1.0, step=0.5)

# Predict Button
if st.button("Calculate Nutrition Facts", type="primary"):
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    
    st.subheader("📊 Estimated Nutrition Results:")
    
    # Columns mein results dikhana
    cols = st.columns(len(target_cols))
    for i, target in enumerate(target_cols):
        label = target.replace('_', ' ').title()
        cols[i].metric(label=label, value=f"{prediction[i]:.1f}")
        
    st.success("Prediction Decision Tree Model ke zariye calculate ki gayi hai!")