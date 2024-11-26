import streamlit as st
import pandas as pd
import pickle
import numpy as np
from io import StringIO
#from dotenv import load_dotenv
from scipy.stats import percentileofscore
import os
from openai import OpenAI
import utils as ut
import requests
from datetime import datetime

client = OpenAI(
  #api provider
  base_url="https://api.groq.com/openai/v1",
  #api_key = os.getenv('GROQ_API_KEY')
  api_key = st.secrets["api_keys"]["GROQ_API_KEY"]
)

current_date = datetime.now()

#define function to load machine learning model
def load_score(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)
  
# Load the original data
def load_original_data():
    url = 'https://raw.githubusercontent.com/AlexandraZavala/Fraud_Detection/main/web/fraudTest.csv'
    
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to load data from GitHub.")
        return None

def prepare_input(customer):
  
  customer['dob'] = pd.to_datetime(customer['dob'])
  age = current_date.year - customer['dob'].year - ((current_date.month, current_date.day) < (customer['dob'].month, customer['dob'].day))
  input_df = {
    "category": customer['category'],
    "age": age,
    "amt": customer['amt'],
    "city": customer['city'],
    "city_pop": customer['city_pop'],
    "state": customer['state'],
    "job": customer['job'],
    "gender": customer['gender'],
  }
  return input_df

def make_predictions(input_df):
    #call the api
    url ="https://fraud-detection-62up.onrender.com"
    #url = "https://localhost:8000"

    try:
        response = requests.post(f"{url}/predict", json=input_df)
        response.raise_for_status()  # Lanza un error si la respuesta no es 200
        result = response.json()
        probabilities = result['probabilities']
        avg_probability = np.mean(list(probabilities.values()))
    except requests.exceptions.RequestException as e:
        print("Error en la solicitud:", e)
        avg_probability = 0  # Asignar un valor por defecto en caso de error

    col1,col2=st.columns(2)
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The transaction has a {avg_probability:.2%} probability of being a fraud")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
  


    return avg_probability

def explain_prediction(probability, input_dict, surname):
    print(probability)
    prompt = f"""
    You are an expert in credit card fraud detection.
    You are given a customer with the following characteristics:
    {input_dict}
    The customer has a {probability:.2%} probability of being a fraudster.

    
    Here are the machine learning model's top 10 most important features for predicting if a transaction is a fraud:

    | Feature    | Importance |
    |------------|------------|
    | amt        | 0.361559   |
    | category   | 0.326383   |
    | gender_F   | 0.124091   |
    | age        | 0.063777   |
    | city_pop   | 0.045933   |
    | job        | 0.028052   |
    | city       | 0.026335   |
    | state      | 0.023871   |
    | gender_M   | 0.000000   |

    Depending of the probability 
    - If the customer has over a 40% of being a fraud, generate a 3 sentence explanation of why the transaction is suspicious.
    - But if the customer has less than a 40% of being a fraud, generate a 3 sentence explanation of why the transaction might not be suspicious.

    The explanation should be in third person, not in first person.
    Don't give information of the customer, just use the surname {surname} of the client of the transaction.
    Don't mention the probability of churning, or the machine learning model, and don't say anything like "Based on the machine learning model's prediction and 10 top most important features", just explain the prediction. Don't mention the importances of the feature.
    Don't repeat information, just explain the prediction.
    Explain the prediction in a way that is easy to understand for a non-expert audience.
    Don't mention transaction ID or related information, just use the surname {surname} of the client of the transaction.
    """
    raw_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{
        "role": "user",
        "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


st.set_page_config(layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.divider()

@st.cache_data
def load_data():
    return load_original_data()


df = load_data()

items_per_page = 120
total_pages = len(df) // items_per_page + (1 if len(df) % items_per_page > 0 else 0)


if 'page' not in st.session_state:
    st.session_state.page = 1

def change_page(delta):
    st.session_state.page = max(1, min(st.session_state.page + delta, total_pages))
    st.rerun()

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if st.button("⬅️ Previous"):
        change_page(-1)
with col2:
   st.markdown(f"<h5 style='text-align: center;'>Page {st.session_state.page} of {total_pages}</h5>", unsafe_allow_html=True)
with col3:
    if st.button("Next ➡️"):
        change_page(1)

current_page = st.session_state.get("page", 1)
start_idx = (current_page - 1) * items_per_page
end_idx = start_idx + items_per_page


columns_to_display = ['trans_date_trans_time','cc_num', 'first', 'last', 'category', 'amt', 'city', 'state', 'job']
df_page = df.iloc[start_idx:end_idx][columns_to_display].copy()



column_configuration = {
    "first": st.column_config.TextColumn(
        "First Name", help="The name of the user", max_chars=100, width="small"
    ),
    "last": st.column_config.TextColumn(
        "Last Name", help="The name of the user", max_chars=100, width="small"
    ),
    "cc_num": st.column_config.TextColumn(
        "Credit Card Number", help="The credit card number of the user", max_chars=150, width="medium"
    ),
    "trans_date_trans_time": st.column_config.TextColumn(
        "Transaction Date", help="The date and time of the transaction", max_chars=100, width="medium"
    ),
    "category": st.column_config.TextColumn(
        "Category", help="The category of the transaction", max_chars=100, width="small"
    ),
    "amt": st.column_config.NumberColumn(
        "Amount", help="The amount of the transaction", width="small"
    ),
    "city": st.column_config.TextColumn(
        "City", help="The city of the transaction", max_chars=100, width="small"
    ),
    "state": st.column_config.TextColumn(
        "State", help="The state of the transaction", max_chars=100, width="small"
    ),
    "job": st.column_config.TextColumn(
        "Job", help="The job of the user", max_chars=100, width="medium"
    ),
}

event = st.dataframe(
    df_page,
    column_config=column_configuration,
    use_container_width=True,
    
    on_select="rerun",
    selection_mode="single-row",
)

if event.selection.rows.__len__() > 0:
    
    
    selected_customer_index_table = event.selection.rows[0]
    
    selected_customer_index = df_page.index[selected_customer_index_table]

    selected_customer = df.iloc[selected_customer_index]

    input_df = prepare_input(selected_customer)

    probability = make_predictions(input_df)

    st.write("Explanation of the prediction:")

    st.divider()

    explanation = explain_prediction(probability, input_df, selected_customer['first'] + ' ' + selected_customer['last'])
    st.write(explanation)

