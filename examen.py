import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificación  ''')
st.image("examen.jpg", caption="Predicción de la calificación de una persona.")

st.header('Datos')

def user_input_features():
  # Entrada
  estudio = st.number_input('Horas de estudio:', min_value=0, max_value=100, value = 0)
  sueño = st.number_input('Horas de estudio:',  min_value=0, max_value=24, value = 0)
  asistencia = st.number_input('Porcentaje de asistencia:', min_value=0, max_value=100, value = 0)
  anteriores = st.number_input('Resultado del examen anterior:', min_value=0, max_value=100, value = 0)


  user_input_data = {'hours_studied': Edad,
                     'sleep_hours': Genero,
                     'attendance_percent': Altura,
                     'previous_scores': Cintura,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

datos =  pd.read_csv('Examscore_df.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613080)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Cálculo de calificación')
st.write('Tu calificación es ', prediccion)
