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


def main():
    st.set_page_config(
        page_title="DermaScan",
        page_icon=":microscope:",
        layout="wide",  # Ancho completo
    )
    st.title("DermaScan")


def pagina_categoria_1():
    st.header("Página 1")
    st.write("Contenido pagina 1.")

if st.button("Home"):
    st.switch_page("src/app.py")
if st.button("DermaScan App"):
    st.switch_page("src/pages/page_1.py")
if st.button("Cuidado y prevención"):
    st.switch_page("src/pages/page_2.py")
if st.button("About us"):
    st.switch_page("src/pages/page_3.py")
if st.button("About us"):
    st.switch_page("src/pages/page_4.py")

'''
    # Crear un menú desplegable en la barra lateral
    categoria_seleccionada = st.sidebar.selectbox("MENÚ", ["Inicio", "DermaScan App", "Prevención", "Índice UV", "About US"])

    # Mostrar la página correspondiente según la categoría seleccionada
    if categoria_seleccionada == "Inicio":
        pagina_categoria_1()
    elif categoria_seleccionada == "DermaScan App":
        pagina_categoria_2()
    elif categoria_seleccionada == "Prevención":
        pagina_categoria_3()
    elif categoria_seleccionada == "Índice UV":
        pagina_categoria_4()
    elif categoria_seleccionada == "About US":
        pagina_categoria_5()
'''




if __name__ == "__main__":
    main()