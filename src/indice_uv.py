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


st.set_page_config(
            page_title="DermaScan",
            page_icon=":☀️:",
            layout="wide",  # Ancho completo
        )

class Indice_UV(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):
        
        st.title("DermaScan")

        st.header("Índice Solar UV: Conoce el Impacto del Sol en tu Piel")
        st.write("¡Descubre el impacto del sol en tu piel con nuestra sección de Incidencia Solar UV! Aquí te proporcionamos información en tiempo real sobre el índice UV actual en diferentes regiones, así como pronósticos para los próximos días. Conoce cómo la radiación UV afecta tu piel en función de la época del año y la latitud en la que te encuentres, y aprende a tomar medidas preventivas para proteger tu piel contra los daños causados por la exposición al sol.")