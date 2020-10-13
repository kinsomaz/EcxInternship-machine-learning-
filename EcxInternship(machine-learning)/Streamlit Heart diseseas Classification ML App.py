
import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import image

pickle_in = open("knn.pk1" , "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome All"
def predict_heart_diseases():
    prediction = classifier.predict([[  ]])
    print(prediction)
    return prediction

def main():
    st.title(" Heart Diseases Classifier ")
    html_temp = """
    <div style = "background-color:tomato ; padding :10px ">
    <h2 style = "color : white ; text-align:center ; "> Streamlit Heart Diseases Classifier ML App </h2>
    </div>
     """
    st.markdown(html_temp, unsafe_allow_html = True)


    results = ""
    if st.button("Predict"):
        result =  predict_heart_diseases()
    st.success('The output is {} '.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if__name == '__main__':
    main()








    
