import streamlit as st
import requests
import asyncio

url = "http://127.0.0.1:8000/"

st.title("Energy predictor from Sample charges and co-ordinates")
st.subheader("Provide charges and co-ordinates : ")
submit = None
with st.form("Form", clear_on_submit=False):
    charges = st.text_input("Enter Charges")
    coord = st.text_input("Enter Co-ordinates")
    submit = st.form_submit_button("Submit Values")

if True:
    requests.post(url+"postCharges", charges)
    requests.post(url+"postCco",coord)

    if st.button('Init Atoms'):
        d1      = requests.get(url+"initAtoms")
    if st.button('Get Values'):
        data    = requests.get(url+"getValues")
        st.write("Values : ")
        st.write(data.content)
    if st.button('Get Charges'):
        charges = requests.get(url+"getCharges")
        st.write("Charges :")
        st.write(charges.content)
    if st.button('Get Forces'):
        forces  = requests.get(url+"getForces")
        st.write("Forces :")
        st.write(forces.content)
    if st.button('Get Potential Energy'):
        pe      = requests.get(url+"getPe")
        st.write("Potential Energy :")
        st.write(pe.content)