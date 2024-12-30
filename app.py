import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle


#load trained model
model = tf.keras.models.load_model('churn_model.h5')

#load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)


#streamlit app
st.title('Customer churn prediction')

#input form
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input('Estimated salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_credit_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

#preprocess input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' :[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded], axis=1)

prediction = model.predict(scaler.transform(input_data))
prediction_probab = prediction[0][0]

if prediction_probab > 0.5:
    st.write('Customer will churn')
else:    
    st.write('Customer will not churn')