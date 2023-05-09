import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

def main():
    #creo le tre variabili di input da inserire
    rd=st.number_input("inserisci il valore di R&D Spend",1,1000,500)
    adm=st.number_input("inserisci il valore di Administration",1,1000,500)
    mkt=st.number_input("inserisci il valore di Marketing_Spend",1,1000,500)
    
    newmodel = joblib.load('regression_test.pkl') ## to load model, carico il modello 
    res=newmodel.predict([[rd,adm,mkt]])[0] #faccio inferenza con il nuovo modello, ossia previsione inserendo 3 input
    st.write(round(res,1)) #stampo il modello



if __name__ == "__main__":
    main()
#scrivere in terminale per lanciare il server : run app_inferenza_9-5.py