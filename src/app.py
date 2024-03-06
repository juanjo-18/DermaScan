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


if __name__ == '__main__':

# Use local CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")    

    
    # ESTA ES LA PAGINA HOST A LA QUE LE AÑADIMOS LAS HIJAS
    app = HydraApp(title='DermaScan',favicon="🔬", hide_streamlit_markers=True,use_navbar=True, navbar_sticky=True)
  
    # AÑADIMOS LAS CLASES
    app.add_app("Inicio", icon="🔬", app=Home())
    app.add_app("Dermascan App",icon="🤳", app=DermascanApp())
    app.add_app("Prevencion", icon="😎", app=Prevencion())
    app.add_app("Indice UV", icon="☀️", app=Indice_UV())
    app.add_app("Sobre nosotros", icon="👥", app=About_US())
   

    # EJECUTA EL MAIN
    app.run()