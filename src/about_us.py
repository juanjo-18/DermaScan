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


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

class About_US(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA
    def run(self):
        
        st.title("DermaScan")

        st.header("Sobre nosotros: Conoce a los Creadores de DermaScan")
        st.write("¡Bienvenido a nuestra sección 'About Us'! Aquí puedes conocer más sobre los creadores detrás de DermaScan: Juanjo Medina y Jesús Cánovas. Nos conocimos mientras realizábamos un Máster en inteligencia artificial y Big Data en el centro integrado de formación profesional Alan Turing, y desde entonces hemos estado trabajando juntos en proyectos innovadores. La idea de desarrollar DermaScan surgió de nuestra pasión por la tecnología y nuestra conciencia sobre los desafíos en la detección temprana del cáncer de piel. Únete a nosotros en nuestra misión de concienciar y prevenir el cáncer de piel mientras exploramos los límites de la inteligencia artificial y la salud digital.")


         # ---- CONTACT ----

        with st.container():
            st.write("---")
            st.header("Get In Touch With Me!")
            st.write("##")

            # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
            contact_form = """
            <form action="https://formsubmit.co/stdust69cleaner@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <textarea name="message" placeholder="Your message here" required></textarea>
                <button type="submit">Send</button>
            </form>
            """
            left_column, right_column = st.columns(2)
            with left_column:
                st.markdown(contact_form, unsafe_allow_html=True)
            with right_column:
                st.empty()
