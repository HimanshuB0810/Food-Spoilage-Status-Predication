import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the scaler ,dataframe and model pickle file

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

with open("dataframe.pkl","rb") as file:
    df=pickle.load(file)

with open("model.pkl","rb") as file:
    model=pickle.load(file)

# Streamlit App
st.title("Food Spoilage Status Predication")

#User input
Ethylene=st.number_input("Ethylene (ppm)")
Co2=st.number_input("CO2 (ppm)")
Temperature=st.number_input("Temperature (C)")
Humidity=st.number_input("Humidity (%RH)")

# Prepare The Input Data
input_data=pd.DataFrame({
    "Ethylene (ppm)":[Ethylene],
    "CO2 (ppm)":[Co2],
    "Temperature (C)":[Temperature],
    "Humidity (%RH)":[Humidity]
})

# Scale the input data
input_data_scaled=scaler.transform(input_data)

# Predict
predication=model.predict(input_data_scaled)

if predication==1:
    st.error("Spoil Status: 1")
else:
    st.success("Spoil Status: 0")



