import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sklearn
sklearn.set_config(transform_output="pandas")

# Загрузка модели из файла
ml_model = joblib.load('/home/saule/ds_bootcamp/Light_GBM/Working_files/ml_pipeline.pkl')
preprocessor = ml_model.named_steps['preprocessor']  # чтобы сделать препроцессинг тестовых данных
# Функция для предсказания цен на недвижимость
def predict_house_prices(data):
    prediction = ml_model.predict(data)
    return prediction

st.title("Предсказание цен на недвижимость")
st.write("Загрузите ваш файл")
uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])

if uploaded_test is not None:
    test = pd.read_csv(uploaded_test)
   # preprocessor.fit_transform(test)  # 
   # predictions = ml_model.predict(test)
    predictions = predict_house_prices(test)
    st.write("Предсказанные цены на недвижимость:")
    st.table(predictions)