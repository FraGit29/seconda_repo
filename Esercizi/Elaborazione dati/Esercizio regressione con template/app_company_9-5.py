import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

def main():
    #creo le tre variabili di input da inserire
    tv=st.number_input("inserisci gli investimenti nella tv",1,1000,150)
    rad=st.number_input("inserisci gli investimenti in radio",1,1000,150)
    nw=st.number_input("inserisci gli investimenti in Newspaper",1,1000,150)
    
    newmodel1 = joblib.load('regression_test_company.pkl') ## to load model, carico il modello 
    res_comp=newmodel1.predict([[tv,rad,nw]])[0] #faccio inferenza con il nuovo modello, ossia previsione inserendo 3 input
    st.write(round(res_comp,1)) #stampo il modello



if __name__ == "__main__":
    main()
#scrivere in terminale per lanciare il server : streamlit run app_company_9-5.py