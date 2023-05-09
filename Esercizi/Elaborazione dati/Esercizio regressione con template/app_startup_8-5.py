import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    #creo le tre variabili di input da inserire
    df=pd.read_csv("https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Startup.csv",sep=",")
    rd=st.number_input("inserisci il valore di R&D Spend",1,1000,500)
    adm=st.number_input("inserisci il valore di Administration",1,1000)
    mkt=st.number_input("inserisci il valore di Marketing_Spend")
    
    X=df.drop(columns='Profit') #tolgo la colonna profit dal df perchè la mia x sarà composta dalle prime tre colonne
    y=df['Profit']#questa sarà la mia y
    model=LinearRegression() #creo un modello con regressione lineare
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 667
                                                    )
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    pd.DataFrame(list(zip(y_pred,y_test)))
    res_df = pd.DataFrame(data=list(zip(y_pred, y_test)),columns=['predicted', 'real'])
    res_df['error'] = res_df['real'] - res_df['predicted']
    length = y_pred.shape[0] #  
    x = np.linspace(0,length,length)

    
    fig=plt.figure(figsize=(10,7))
    plt.plot(x, y_test, label='real y')
    plt.plot(x, y_pred, label="predicted y'")
    plt.legend(loc=2)
    st.pyplot(fig)
    
    '''Il primo pezzo di codice assegna alla variabile "length" la lunghezza della prima dimensione dell'array "y_pred" (ovvero il numero di righe 
presenti nell'array).
Il secondo pezzo di codice crea un array di numeri equidistanti che ha come lunghezza proprio quella assegnata in precedenza alla variabile
 "length", e tale array viene poi assegnato alla variabile "x". In altre parole, "x" sarà un array di numeri che vanno da 0 a "length-1", 
 con passo 1.'''
    from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error 
    r2score= r2_score(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)
    mse=mean_squared_error(y_test, y_pred)
    rmse=mean_squared_error(y_test,y_pred,squared=False)

    st.write(f'R2_score:{r2score}')
    st.write('MAE:',mae)
    st.write('MSE:',mse)
    st.write('RMSE',rmse)

    newmodel = joblib.load('regression_test.pkl') ## to load model, carico il modello 
    res=newmodel.predict([[rd,adm,mkt]])[0] #faccio inferenza con il nuovo modello, ossia previsione inserendo 3 input
    st.write(round(res,1)) #stampo il modello



if __name__ == "__main__":
    main()
#scrivere in terminale per lanciare il server : run app_inferenza_9-5.py