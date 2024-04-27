import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Анализ и подготовка данных к созданию модели')

st.write('Данные для обучения модели')
train = pd.read_csv('/home/saule/ds_bootcamp/Light_GBM/Data/train.csv')
st.table(train.head(4))

st.write('NaN значения в датафрейме')
empty = pd.DataFrame(data={'NaN_count': train.isna().sum(), 'data_type':train.dtypes})
empty = empty[empty['NaN_count'] > 0]
st.table(empty)


st.write('В обучающей выборки убирали элементы с SalePrice выше 300 000')
fig, ax = plt.subplots(figsize=(16, 8))
plt.bar(train.index, train['SalePrice'])
ax.axhline(y=300000, color='red')
ax.set_title('График цен')
st.pyplot(fig)

st.write('В обучающей выборки убирали элементы с TotalBsmtSF (площадь) более 2000')
fig, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(data=train, x='TotalBsmtSF', y='SalePrice')
ax.axvline(x=2000, color='red')
st.pyplot(fig)

st.write('Столбцы с признаками, не имеющими значения, удаляли полностью')
fig, ax = plt.subplots(figsize=(16, 8))
plt.hist(train['ExterCond'])
ax.set_title('Гистограмма распределения значений параметра ExterCond')
st.pyplot(fig)