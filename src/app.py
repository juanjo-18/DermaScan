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

PINNED_NAV_STYLE = """
                    <style>
                    .reportview-container .sidebar-content {
                        padding-top: 0rem;
                    }
                    .reportview-container .main .block-container {
                        padding-top: 1rem;
                        padding-right: 1rem;
                        padding-left: 1rem;
                        padding-bottom: 0rem;
                    }
                    </style>
                """

if __name__ == '__main__':

    # ESTA ES LA PAGINA HOST A LA QUE LE AÑADIMOS LAS HIJAS
    app = HydraApp(title='DermaScan',favicon="🔬")
  
    # AÑADIMOS LAAS CLASES
    app.add_app("Inicio", icon="🔬", app=Home())
    app.add_app("Dermascan App",icon="🤳", app=DermascanApp())
    app.add_app("Prevencion", icon="😎", app=Prevencion())
    app.add_app("Indice UV", icon="☀️", app=Indice_UV())
    app.add_app("Sobre nosotros", icon="👥", app=About_US())
   
    st.markdown(PINNED_NAV_STYLE,unsafe_allow_html=True)
    # EJECUTA EL MAIN
    app.run()