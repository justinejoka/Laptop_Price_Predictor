import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# Load the Pipeline using the same version of scikit-learn used for saving it
with open('pipe.pkl', 'rb') as file1:
    rf = joblib.load(file1)

# Check the scikit-learn version
print(sklearn.__version__)

# Apple,Ultrabook,8,Mac,1.37,0,1,226.98300468106115,Intel Core i5,0,128,Intel

data = pd.read_csv("traineddata.csv")

data['IPS'].unique()

st.title("Laptop Price Predictor")

company = st.selectbox('Brand', data['Company'].unique())

# type of laptop
type = st.selectbox('Type', data['TypeName'].unique())

# Ram present in laptop
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# os of laptop
os = st.selectbox('OS', data['OpSys'].unique())

# weight of laptop
mean_weight = data['Weight'].mean()
data['Weight'] = data['Weight'].replace('1.37', mean_weight)

weight = st.selectbox('Weight of the laptop', data['Weight'].unique())

# touchscreen available in laptop or not
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.selectbox('Screen Size', data['PPI'].unique())

# resolution of laptop
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900',
                          '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                          '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

# Define help section
with st.expander("ℹ️ Help"):
    st.write("This app predicts the price of a laptop based on its specifications.")
    st.write("Please select the laptop brand, type, RAM, OS, weight, touchscreen availability, IPS display, screen size, screen resolution, CPU, HDD, SSD, and GPU from the dropdown menus.")
    st.write("Then click on the 'Predict Price' button to see the predicted price range for the laptop.")

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])




    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = np.array([company, type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    prediction = int(np.exp(rf.predict(query)[1]))
    exchange_rate = 0.014
    prediction_usd = round(prediction * exchange_rate, 2)

    st.title("Predicted price for this laptop could be between " +
             str(prediction_usd - 14.00) + "$" + " to " + str(prediction_usd + 14.00) + "$")