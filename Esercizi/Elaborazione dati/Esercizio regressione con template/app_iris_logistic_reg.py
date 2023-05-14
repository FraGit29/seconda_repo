import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

def main():
    #creo le tre variabili di input da inserire
    sl=st.number_input("inserisci il valore di Sepal lenght",1,10,3)
    sw=st.number_input("inserisci il valore di Sepal width",1,10,3)
    pl=st.number_input("inserisci il valore di Petal lenght",1,10,3)
    pw=st.number_input("Inserisci il valore di Petal Width",1,10,1)
    newmodel = joblib.load('logistic_regression_test.pkl') ## to load model, carico il modello 
    res=newmodel.predict([[sl,sw,pl,pw]])[0] #faccio inferenza con il nuovo modello, ossia previsione inserendo 3 input
    if st.button('prediction'):
        st.write(res) #stampo il modello



if __name__ == "__main__":
    main()
#scrivere in terminale per lanciare il server : run app_inferenza_9-5.py