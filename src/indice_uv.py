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
from unidecode import unidecode
from bs4 import BeautifulSoup


class Indice_UV(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):

        col1, col2, col3 = st.columns([2, 6, 2])

        with col2:
            st.markdown(f"<h1 style='text-align:center;'>Índice Solar UV: Conoce el Impacto del Sol en tu Piel</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center;'>Consulta el Índice UV en las principales capitales de provincia de España</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Aquí puedes encontrar información en tiempo real sobre el índice UV actual en diferentes regiones, así como pronósticos para los próximos días. Conoce cómo la radiación UV afecta tu piel en función de la época del año y la latitud en la que te encuentres.</p>", unsafe_allow_html=True)
            
            st.write("")
            # Saco la pagina
            # Creo la tabla
            # Lista de nombres
            nombres = ["Barcelona", "Bilbao","Granada","Madrid","Málaga","Sevilla","Valencia","Zaragoza"]

            # Crear un desplegable con st.multiselect
            nombres_seleccionados = st.selectbox("Selecciona una provincia:", nombres)

            # Mostrar los nombres seleccionados
            nombre_formateado = unidecode(nombres_seleccionados.lower())

            url = 'https://www.tutiempo.net/'+nombre_formateado+'.html?datos=detallados'

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
                radiacion_uva = contenedor_producto.find_all('span', class_= ['c2', 'c3'])

                for indice, salida in enumerate(dia_texto):
                    for indice2, salida2 in enumerate(dia_numero):
                        for indice3, salida3 in enumerate(temperatura_maxima):
                            for indice4, salida4 in enumerate(temperatura_minima):
                                for indice5, salida5 in enumerate(radiacion_uva):
                                    if indice==indice2==indice3==indice4==indice5:
                                        datos_productos.append({
                                            'Fecha': salida.get_text(),
                                            'Día': salida2.get_text(),
                                            'Radiación UV': salida5.get_text(),
                                            'Temperatura MAX': salida3.get_text(),
                                            'Temperatura MIN': salida4.get_text()
                                            
                                            })


            # Convierte la lista de datos a un DataFrame
            df = pd.DataFrame(datos_productos)
            # Mostrar la tabla en Streamlit
            st.table(df.style.set_table_styles([dict(selector="th", props=[("text-align", "center")])]))
        
            
            with st.container():

                st.bar_chart(df, y="Día", x="Radiación UV", color="Radiación UV")

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
                    st.empty()
                    #st.image("imagenes/imagen_radiacion_uva.png", use_column_width=True)
                    #st.markdown(f'<img src="./app/static/imagenes/imagen_radiacion_uva.png" height="" style="{style_image2}">', unsafe_allow_html=True)
                with col3:
                    st.empty()

            
            st.divider()