import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="PrachiRB/Tourism_prediction_model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness tourism package prediction App")
st.write("""
This app predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
 Age = st.number_input("Age", min_value=0, max_value=100, value=30)
 Number_Of_Person_Visiting = st.number_input("Number of Person Visiting", min_value=0, max_value=20, value=2)
 Number Of Children Visiting less than 5years = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
 Preferred_Property_Star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
 Average Number Of Trips in a year = st.number_input("Average Number Of Trips in a year", min_value=0, max_value=10, value=2)
 Do the customer have a valid Passport ? = st.selectbox("Passport",["Yes","No"])
 Do the customer have his/her own Car ? = st.selectbox("OwnCar",["Yes","No"])
 Pitch Satisfaction Score = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=5)
 Number Of Followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=2)
 Type of Contact = st.selectbox("Type of Contact",["Customer Invited","Self Inquiry"])
 City Tier = st.selectbox("City Tier",["Tier 1","Tier 2","Tier 3"] 
 Occupation = st.selectbox("Occupation",["Self-Employed","Salaried","Business"],"Freelancer")
 Gender = st.selectbox("Gender",["Male","Female"])
 Marital_Status = st.selectbox("Marital Status",["Married","Single","Divorced"])
 Designation = st.selectbox("Designation",["Executive","Managerial","Professional","Self-Employed","Other"])
 Product Pitched = st.selectbox( "Basic","Standard","Deluxe","Super Deluxe")
 Number Of Followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=2)
 Duration Of Pitch(days) = st.number_input("Duration Of Pitch", min_value=0, max_value = 50, value=5)



# Assemble input into DataFrame
input_data = pd.DataFrame([{
   'Age': Age,
    'NumberOfPersonVisiting': Number_Of_Person_Visiting,
    'NumberOfChildrenVisiting': Number Of Children Visiting less than 5years,
    'PreferredPropertyStar': Preferred_Property_Star,
    'NumberOfTrips': Average Number Of Trips in a year
    'Passport': Do the customer have a valid Passport ?
    'OwnCar': Do the customer have his/her own Car ?
    'PitchSatisfactionScore': Pitch Satisfaction Score
    'NumberOfFollowups': Number Of Followups
    'TypeofContact': Type Of Contact
    'CityTier': City Tier
    'Occupation': Occupation
    'Gender': Gender
    'MaritalStatus': Marital_Status
    'Designation': Designation
    'ProductPitched': Product Pitched
}])


if st.button("Prod Taken"):
    prediction = model.predict(input_data)[0]
    result = "Prod Taken" if prediction == 1 else "Prod Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
