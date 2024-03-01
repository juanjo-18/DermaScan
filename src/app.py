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

from home import Home
from dermascan_app import DermascanApp
from prevencion import Prevencion
from indice_uv import Indice_UV
from about_us import About_US
import streamlit.components.v1 as components
from hydralit_components import IS_RELEASE



if __name__ == '__main__':

    st.set_page_config(
        page_title="Inicio",
        page_icon=":🔬:",
        layout="wide",  # Ancho completo
    )

    # ESTA ES LA PAGINA HOST A LA QUE LE AÑADIMOS LAS HIJAS
    app = HydraApp(title='DermaScan',favicon="🔬")
  
    # AÑADIMOS LAAS CLASES
    app.add_app("Inicio", icon="🔬", app=Home())
    app.add_app("Dermascan App",icon="🤳", app=DermascanApp())
    app.add_app("Prevencion", icon="😎", app=Prevencion())
    app.add_app("Indice UV", icon="☀️", app=Indice_UV())
    app.add_app("Sobre nosotros", icon="👥", app=About_US())
   
    css = """
    <style>
        #custom-app {
            position: fixed;
            top: 0;
            width: 100%;
        }
    </style>
    """
    css_2 = """
    <style>
        body {
            width: 100%;
            height: 100%
        }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
    st.markdown(css_2, unsafe_allow_html=True)

    # EJECUTA EL MAIN
    app.run()