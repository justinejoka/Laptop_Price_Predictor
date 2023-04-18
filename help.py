import streamlit as st

def app():
    st.title("Help")
    st.write("Welcome to the help page!")
    st.write("To use this app, please select the values for the following features:")
    st.write("- Brand")
    st.write("- Type")
    st.write("- RAM")
    st.write("- OS")
    st.write("- Weight")
    st.write("- Touchscreen")
    st.write("- IPS")
    st.write("- Screen Size")
    st.write("- Screen Resolution")
    st.write("- CPU")
    st.write("- HDD")
    st.write("- SSD")
    st.write("- GPU")
    st.write("")
    st.write("After selecting the values, click on the 'Predict Price' button to get the predicted price of the laptop.")