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
    page_icon=":👥:",
    layout="wide",  # Ancho completo
)
st.title("DermaScan")

st.header("About Us: Conoce a los Creadores de DermaScan")
st.write("¡Bienvenido a nuestra sección 'About Us'! Aquí puedes conocer más sobre los creadores detrás de DermaScan: Juanjo Medina y Jesús Cánovas. Nos conocimos mientras realizábamos un Máster en inteligencia artificial y Big Data en el centro integrado de formación profesional Alan Turing, y desde entonces hemos estado trabajando juntos en proyectos innovadores. La idea de desarrollar DermaScan surgió de nuestra pasión por la tecnología y nuestra conciencia sobre los desafíos en la detección temprana del cáncer de piel. Únete a nosotros en nuestra misión de concienciar y prevenir el cáncer de piel mientras exploramos los límites de la inteligencia artificial y la salud digital.")
    
