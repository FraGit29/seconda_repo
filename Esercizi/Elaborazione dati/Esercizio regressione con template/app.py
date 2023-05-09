import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
def main():
    st.write("testo prova")

    sl1=st.slider("Inserire il numero di punti",1,100,70)
    sl2=st.slider("Inserire il coeff angolare",1,10,5)
    sl3=st.slider("Inserire il valore della SD",1,10,3)
    '''con questi slider genero una barra con cui puoi 
    modificare i punti in tempo reale (simile a insert), con min, max e default'''
    generate_random = np.random.RandomState(667) #667 random valido per tutti
    x = 10 * generate_random.rand(sl1) #creiamo un valore di x random moltiplicato per 10
    noise = np.random.normal(0,1,sl1) #random da 0 a 100 con media 0 e dev std 1
    y = noise + sl2*x

    fig=plt.figure(figsize=(12,10))#inizializza una figura vuota
    plt.plot(x,y,'o') #la riempiamo con un plot
    plt.axis([0,10,0,30])#misure degli assi

   

    
    st.pyplot(fig) #la renderizziamo a video






if __name__ == "__main__":
    main()
#scrivere in terminale per lanciare il server : streamlit run app.py 