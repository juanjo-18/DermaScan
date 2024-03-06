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

   
    # ESTA ES LA PAGINA HOST A LA QUE LE AÑADIMOS LAS HIJAS
    app = HydraApp(title='DermaScan',favicon="🔬", hide_streamlit_markers=True,use_navbar=True, navbar_sticky=True)
  
    # AÑADIMOS LAS CLASES
    app.add_app("INICIO", icon="🔬", app=Home())
    app.add_app("DERMASCAN APP",icon="🤳", app=DermascanApp())
    app.add_app("PREVENCIÓN", icon="😎", app=Prevencion())
    app.add_app("ÍNDICE UV", icon="☀️", app=Indice_UV())
    app.add_app("SOBRE NOSOTROS", icon="👥", app=About_US())
   

    # EJECUTA EL MAIN
    app.run()