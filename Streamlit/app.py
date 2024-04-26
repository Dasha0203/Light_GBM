import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Загрузка модели из файла
ml_model = joblib.load('/home/saule/ds_bootcamp/Light_GBM/Working_files/ml_pipeline.pkl')
preprocessor = ml_model.named_steps['preprocessor']  # чтобы сделать препроцессинг тестовых данных
# Функция для предсказания цен на недвижимость
#def predict_house_prices(model, data):
   # prediction = model.predict(data)
    #return prediction

st.title("Предсказание цен на недвижимость")

uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])

if uploaded_test is not None:
    test = pd.read_csv(uploaded_test)
    preprocessor.fit_transform(test)  # 
    predictions = ml_model.predict(test)
    st.write("Предсказанные цены на недвижимость:")
    st.write(predictions)