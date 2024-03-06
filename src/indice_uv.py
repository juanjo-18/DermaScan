import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import tensorflow
import keras
import boto3
import os

from keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
from st_files_connection import FilesConnection
from botocore.exceptions import NoCredentialsError
from hydralit import HydraApp
from hydralit import HydraHeadApp
import requests
from bs4 import BeautifulSoup


class Indice_UV(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):

        st.header("Índice Solar UV: Conoce el Impacto del Sol en tu Piel")
        st.write("¡Descubre el impacto del sol en tu piel con nuestra sección de Incidencia Solar UV! Aquí te proporcionamos información en tiempo real sobre el índice UV actual en diferentes regiones, así como pronósticos para los próximos días. Conoce cómo la radiación UV afecta tu piel en función de la época del año y la latitud en la que te encuentres, y aprende a tomar medidas preventivas para proteger tu piel contra los daños causados por la exposición al sol.")
        # Saco la pagina
        # Creo la tabla
        url = 'https://www.tutiempo.net/malaga.html?datos=detallados'

        response = requests.get(url)

        # Verifica si la solicitud fue exitosa (código de estado 200)
        if response.status_code == 200:
            # Obtén el contenido HTML como texto
            contenido_html = response.text

        # Lista para almacenar los datos
        datos_productos = []
        soup = BeautifulSoup(contenido_html, 'html.parser')

        # Busco los contenedores donde se encuentran los datos que busco
        contenedores_productos =  soup.find_all('div', id='ColumnaIzquierda')

        # Recorro la lista completa
        for contenedor_producto in contenedores_productos:
            # Busca el dia texto
            dia_texto = contenedor_producto.find_all('span', class_='day')
            # Busca el dia numero
            dia_numero = contenedor_producto.find_all('h3')
            # Busca la temperatura maxima
            temperatura_maxima = contenedor_producto.find_all('span', class_='t max')
            # Busca la temperatura minima
            temperatura_minima = contenedor_producto.find_all('span', class_='t min')
            # Busca la radiacion uva
            radiacion_uva = contenedor_producto.find_all('span', class_='c2')

            for indice, salida in enumerate(dia_texto):
                for indice2, salida2 in enumerate(dia_numero):
                    for indice3, salida3 in enumerate(temperatura_maxima):
                        for indice4, salida4 in enumerate(temperatura_minima):
                            for indice5, salida5 in enumerate(radiacion_uva):
                                if indice==indice2==indice3==indice4==indice5:
                                    datos_productos.append({
                                        'Dia': salida.get_text(),
                                        'Dia de la semana': salida2.get_text(),
                                        'Temperatura MAX': salida3.get_text(),
                                        'Temperatura MIN': salida4.get_text(),
                                        'Radiación UV': salida5.get_text()
                                        })


        # Convierte la lista de datos a un DataFrame
        df = pd.DataFrame(datos_productos)

        # Mostrar la tabla en Streamlit
        st.table(df.style.set_table_styles([dict(selector="th", props=[("text-align", "center")])]))
       
        
        with st.container():

            st.area_chart(data=df, x='Dia de la semana', y='Radiación UV', color="#ffaa0088", width=0, height=0, use_container_width=True)


            # You can call any Streamlit command, including custom components:
            #st.bar_chart(df)
            #chart_data = df(('Dia de la semana', 'Temperatura MAX'), columns=["Radiación UV"])
            #st.line_chart(chart_data)

        with st.container():
            style_image2 = """
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 1000px;
            max-height: 1000px;
            justify-content: center;
            border-radius: ;
            """

            col1, col2, col3 = st.columns(3)
            with col1:
                st.empty()
            with col2:
                st.image("imagenes/imagen_radiacion_uva.png", use_column_width=True)
                #st.markdown(f'<img src="./app/static/imagenes/imagen_radiacion_uva.png" height="" style="{style_image2}">', unsafe_allow_html=True)
            with col3:
                st.empty()

        
      