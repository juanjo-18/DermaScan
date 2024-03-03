import streamlit as st
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
        
        st.title("DermaScan")

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
                                        'Radiación UVA': salida5.get_text()
                                        })


        # Convierte la lista de datos a un DataFrame
        df = pd.DataFrame(datos_productos)

        # Mostrar la tabla en Streamlit
        st.table(df.style.set_table_styles([dict(selector="th", props=[("text-align", "center")])]))
       
        # Puedes proporcionar la ruta de la imagen localmente o una URL
        ruta_imagen = "imagenes/imagen_radiacion_uva.png"  

        # Mostrar la imagen en Streamlit
        st.image(ruta_imagen, caption='', use_column_width=False, width=750)
        st.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="{ruta_imagen}" style="object-fit: contain;" width="750"/>'
            f'</div>',
            unsafe_allow_html=True
        )
        