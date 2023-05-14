import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    st.write('ciao')


    df=pd.read_csv("https://frenzy86.s3.eu-west-2.amazonaws.com/python/penguins.csv",sep=",")
    df=pd.get_dummies(df,columns=['island','sex'])
    X=df.drop(columns='species')
    y=df['species']
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.30, 
                                                    random_state = 667
                                                    )
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    labels = ['Adelie ', 'Chinstrap', 'Gentoo']
    

    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure()
    sns.heatmap(cm , square=True, annot=True, cbar=False,cmap='Blues',xticklabels=labels,yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot(fig)

    st.write(f"L'accuracy totale è:{round(acc,2)*100}%")

if __name__ == "__main__":
    main()
#scrivere in terminale per lanciare il server : run app_inferenza_9-5.py