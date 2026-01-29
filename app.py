import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

st.image('src/Customer_Churn_image.png', width=700)

# Page setup
st.set_page_config(
page_title="Customer Churn Prediction",
layout="wide", # or "wide"
)

# Load trained ANN model
model = tf.keras.models.load_model("saved_model/churn_model.h5")

# Load preprocessor (ColumnTransformer pipeline)
with open("saved_model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

def predict_churn(input_data):
    # Apply SAME preprocessing as training
    input_processed = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(input_processed)
    prediction_proba = prediction[0][0]

    st.subheader(f"Churn Probability: {prediction_proba:.2f}  ")

    if prediction_proba > 0.5:
        st.error("The customer is likely to churn")
    else:
        st.success("The customer is not likely to churn")

# Streamlit UI
st.header(":green[Input Customer Details]")
c0,c1 = st.columns(2)
with c0:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    has_cr_card_str = st.radio("Has Credit Card", ['No', 'Yes'])
    is_active_member_str = st.radio("Is Active Member", ['No', 'Yes'])


with c1:
    balance = st.number_input("Balance", value=60000.0)
    estimated_salary = st.number_input("Estimated Salary", value=50000.0)
    age = st.slider("Age", 18, 92, 25)
    tenure = st.slider("Tenure", 0, 10, 3)
    num_of_products = st.slider("Number of Products", 1, 4, 1)

agreement = st.checkbox(":orange[I confirm that the above information is correct.]")
        
if st.button("Predict Churn"):
    if agreement:
        has_cr_card = 1 if has_cr_card_str == 'Yes' else 0
        is_active_member = 1 if is_active_member_str == 'Yes' else 0
        # Prepare input DataFrame
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        predict_churn(input_data)
    else:
        st.warning("Please confirm that the information is correct by checking the box.")
