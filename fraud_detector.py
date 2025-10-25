import streamlit as st
import joblib
import pandas as pd
import re
from rapidfuzz import fuzz

st.set_page_config(
    page_title="Bank Fraud Detector",
    page_icon="🛡️", 
    layout="centered"
)

st.title("Bank Account Application Fraud Detector")

@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    return model

model = load_model()

@st.cache_resource
def load_encoder():
    le = joblib.load('label_encoder.pkl')
    return le

le = load_encoder()

st.subheader("Option 1: Check Individual (please fill in the required fields)")

def similarity(name, email):
    name = re.sub(r'[^a-zA-Z]', '', name.lower())
    email_user = re.sub(r'[^a-zA-Z]', '', email.split('@')[0].lower())
    
    ratio = fuzz.ratio(name, email_user) / 100.0
    return ratio

income = st.number_input("Income ($)")
income = min((income / 150000), 1)
name = st.text_input("Name")
email = st.text_input("Email")
name_email_similarity = similarity(name, email)
prev_address_months = st.number_input("Number of months in previous address (-1 if N/A)")
current_address_months = st.number_input("Number of months in current address")
age = st.number_input("Age")
age = (age // 10) * 10
days_since_request = st.number_input("Number of days since application was sent")
intended_balcon = st.number_input("Initial transferred amount for application")
zip_count_4w = st.number_input("Number of applicants with same zip code in last 4 weeks")
velocity_6h = st.number_input("Average number of applications per hour in last 6 hours")
velocity_24h = st.number_input("Average number of applicants per hour in last 24 hours")
bank_branch_count_8w = st.number_input("Total number of applications received by bank branch in last 8 weeks")
dob_distinct_emails_4w = st.number_input("Number of emails for applicants with same date of birth in last 4 weeks")
employment_status = st.text_input("Employment status (CA for employed / CB for unemployed / Other)")
email_free = st.number_input("Is email free? 1 for yes / 0 for no")
housing_status = st.text_input("Housing status (BC for has residence / BB for not / Other)")
phone_home_valid = st.number_input("Is home phone valid? 1 for yes / 0 for no")
phone_mobile_valid = st.number_input("Is mobile phone valid? 1 for yes / 0 for no")
bank_months = st.number_input("How old is previous account? (-1 if N/A)")
has_other_cards = st.number_input("Does applicant have other cards in the same banking company? 1 for yes / 0 for no")
proposed_credit_limit = st.number_input("Applicant's proposed credit limit")
foreign_request = st.number_input("Is request's origin different from bank's country? 1 for yes / 0 for no")
source = st.selectbox("Source of application", ["INTERNET", "TELEAPP"])
session_length_min = st.number_input("Length of user session in banking website (minutes)")
device_os = st.selectbox("Device OS", ["windows", "linux", "macintosh", "other"])
keep_alive_session = st.number_input("Did user choose to keep session alive on logout? 1 for yes / 0 for no")
device_distinct_emails_8w = st.number_input("Number of distinct emails used in banking website from the same device in last 8 weeks")

if st.button("Check"):
    input_df = pd.DataFrame([{
        "income": income,
        "name_email_similarity": name_email_similarity,
        "prev_address_months_count": prev_address_months,
        "current_address_months_count": current_address_months,
        "customer_age": age,
        "days_since_request": days_since_request,
        "intended_balcon_amount": intended_balcon,
        "zip_count_4w": zip_count_4w,
        "velocity_6h": velocity_6h,
        "velocity_24h": velocity_24h,
        "bank_branch_count_8w": bank_branch_count_8w,
        "date_of_birth_distinct_emails_4w": dob_distinct_emails_4w,
        "employment_status": employment_status,
        "email_is_free": email_free,
        "housing_status": housing_status,
        "phone_home_valid": phone_home_valid,
        "phone_mobile_valid": phone_mobile_valid,
        "bank_months_count": bank_months,
        "has_other_cards": has_other_cards,
        "proposed_credit_limit": proposed_credit_limit,
        "foreign_request": foreign_request,
        "source": source,
        "session_length_in_minutes": session_length_min,
        "device_os": device_os,
        "keep_alive_session": keep_alive_session,
        "device_distinct_emails_8w": device_distinct_emails_8w
    }])
    for col, encoder in le.items():
        input_df[col] = encoder.transform(input_df[col])
    result = model.predict(input_df)[0]
    st.success(f"Prediction: {'FRAUD' if result==1 else 'LEGIT'}")

st.subheader("Option 2: Batch Check (please upload a CSV file with all the required fields below filled in for each applicant)")
st.text("Required fields: income, name, email, prev_address_months_count, current_address_months_count, customer_age, days_since_request, intended_balcon_amount, zip_count_4w, velocity_6h, velocity_24h, bank_branch_count_8w, date_of_birth_distinct_emails_4w, employment_status, email_is_free, housing_status, phone_home_valid, phone_mobile_valid, bank_months_count, has_other_cards, proposed_credit_limit, foreign_request, source, session_length_in_minutes, device_os, keep_alive_session, device_distinct_emails_8w")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

def normalize_income(row):
    income = min((row["income"] / 150000), 1)
    return income

def similarity_batch(row):
    name = re.sub(r'[^a-zA-Z]', '', row["name"].lower())
    email_user = re.sub(r'[^a-zA-Z]', '', row["email"].split('@')[0].lower())
    
    ratio = fuzz.ratio(name, email_user) / 100.0
    return ratio

if uploaded and st.button("Check Batch"):
    inputs = pd.read_csv(uploaded)
    inputs["income"] = inputs.apply(normalize_income, axis=1)
    inputs["name_email_similarity"] = inputs.apply(similarity_batch, axis=1)
    inputs.drop("name", axis=1, inplace=True)
    inputs.drop("email", axis=1, inplace=True)
    for col, encoder in le.items():
        inputs[col] = encoder.transform(inputs[col])
    results = model.predict(inputs)
    inputs['Prediction'] = ['FRAUD' if r==1 else 'LEGIT' for r in results]
    st.dataframe(inputs)