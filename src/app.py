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




st.set_page_config(
    page_title="DermaScan",
    page_icon=":microscope:",
    layout="wide",  # Ancho completo
)
st.title("DermaScan")

    
# Encabezado principal
st.header("¡Bienvenidos a DermaScan!")
st.subheader("Descubre el Futuro del Cuidado de la Piel")

col1, col2 = st.columns([1, 1])
with col1:
# Sección: DermaScan App
    st.subheader("DermaScan App: Tu Aliado en la Lucha Contra el Cáncer de Piel")
    st.write("DermaScan App representa la vanguardia en el cuidado de la piel, utilizando modelos de inteligencia artificial para analizar imágenes y detectar posibles lesiones cutáneas. Nuestra aplicación puede identificar si una lesión es maligna, así como clasificarla según su tipo, proporcionando a los usuarios una herramienta poderosa para la detección temprana y la prevención del cáncer de piel.")

with col2:
# Sección: Prevención y cuidado de la piel
    st.subheader("Prevención y Cuidado de la Piel: Tu Guía hacia una Piel Saludable")
    st.write("En esta sección, encontrarás una amplia gama de recomendaciones y consejos sobre cómo mantener tu piel saludable y protegida contra los daños solares. Desde prácticas diarias de cuidado de la piel hasta medidas preventivas contra el cáncer de piel, estamos aquí para ayudarte a mantener una piel radiante y saludable en todo momento.")

col3, col4 = st.columns([1, 1])
with col3:
# Sección: Incidencia Solar UV
    st.subheader("Incidencia Solar UV: Conoce el Impacto del Sol en tu Piel")
    st.write("Descubre la importancia de estar conscienciado del índice UV y su impacto en tu piel. En esta sección, te proporcionamos información en tiempo real sobre el índice UV actual en diferentes regiones, así como pronósticos para los próximos días. Mantente informado y protege tu piel contra los daños causados por la radiación solar.")

with col4:
# Sección: About us
    st.subheader("About us: Conoce a los Creadores de DermaScan")
    st.write("Somos Juanjo Medina y Jesús Cánovas, dos apasionados del aprendizaje automático y la ciencia de datos. Nuestra inspiración para desarrollar DermaScan surgió de la necesidad de crear un modelo de inteligencia artificial para la detección temprana de enfermedades de la piel. Como residentes de Málaga, somos conscientes de la importancia de proteger la piel contra la radiación solar UV, lo que nos impulsó a crear esta aplicación innovadora como nuestro proyecto final de máster. Únete a nosotros en nuestra misión sobre el concienciamiento y la precvención del cáncer de piel.")

