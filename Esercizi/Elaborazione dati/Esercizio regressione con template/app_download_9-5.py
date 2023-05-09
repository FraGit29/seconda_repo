import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import io

def main():
    buffer = io.BytesIO()
    #devo caricare il csv
    df=pd.read_csv("https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Startup.csv",sep=",")
    X=df.drop(columns='Profit')
    st.write("database di input")
    st.write(X)
    newmodel = joblib.load('regression_test.pkl')
    output = newmodel.predict(X)
    st.write("database di output")
    st.write(output)
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()

        download2 = st.download_button(
            label="Download data as Excel",
            data=buffer,
            file_name='large_df.xlsx',
            mime='application/vnd.ms-excel'
        )


if __name__ == "__main__":
    main()
