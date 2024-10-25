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
    #url = 'https://raw.githubusercontent.com/AlexandraZavala/ChurnCustModels/main/web/churn.csv'
    
    #response = requests.get(url)
    #if response.status_code == 200:
    #    return pd.read_csv(StringIO(response.text))
    #else:
    #    st.error("Failed to load data from GitHub.")
    #

    df = pd.read_csv('fraudTrain.csv')

    return df

def prepare_input(customer):
  
  customer['dob'] = pd.to_datetime(customer['dob'])
  age = current_date.year - customer['dob'].year - ((current_date.month, current_date.day) < (customer['dob'].month, customer['dob'].day))
  input_df = {
    "category": customer['category'],
    "amt": customer['amt'],
    "city": customer['city'],
    "state": customer['state'],
    "job": customer['job'],
    "gender_F": 1 if customer['gender'] == 'F' else 0,
    "gender_M": 1 if customer['gender'] == 'M' else 0,
    "age": age,
    "city_pop": customer['city_pop'],
    "AmtAgeRatio": customer['amt'] / age,
  }
  return input_df

def make_predictions(input_df):
  #call the api
  url ="https://churncustmodels-1.onrender.com"
  #url = "https://localhost:8000"


  response = requests.post(f"{url}/predict", json=input_df)
  if response.status_code == 200:
    result = response.json()
  else:
    print("Error:", response.status_code, response.text)
  
  probabilities = result['probabilities']
  avg_probability=np.mean(list(probabilities.values()))

  col1,col2=st.columns(2)
  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)
  
  #customer percentiles chart
  metrics = ['amt', 'age', 'city_pop']
  
  percentiles = {}
  for metric in metrics:
    # Calcular el percentil del cliente seleccionado con respecto a los demás en esa métrica
    percentiles[metric] = percentileofscore(df[metric], selected_customer[metric]) / 100

  fig_percentiles = ut.create_percentiles_chart(percentiles, metrics)
  st.plotly_chart(fig_percentiles, use_container_width=True)

  return avg_probability

def explain_prediction(probability, input_dict, surname):
    prompt = f"""
    You are an expert in credit card fraud detection.
    You are given a customer with the following characteristics:
    {input_dict}
    The customer has a {probability:.2%} probability of being a fraudster.
    Please explain why the model made this prediction.
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

#linea
st.divider()

@st.cache_data
def load_data():
    return pd.read_csv('fraudTest.csv')


df = load_data()

items_per_page = 100
total_pages = len(df) // items_per_page + (1 if len(df) % items_per_page > 0 else 0)


if 'page' not in st.session_state:
    st.session_state.page = 1

# Función para cambiar de página
def change_page(delta):
    st.session_state.page = max(1, min(st.session_state.page + delta, total_pages))
    st.rerun()
# Interfaz de navegación
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
    if st.button("Predict"):
    
        selected_customer_index_table = event.selection.rows[0]
        
        selected_customer_index = df_page.index[selected_customer_index_table]

        selected_customer = df.iloc[selected_customer_index]

        input_df = prepare_input(selected_customer)

        make_predictions(input_df)

