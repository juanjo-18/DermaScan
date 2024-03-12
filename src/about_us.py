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



class About_US(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA
    def run(self):
        
        
        st.markdown(f"<h1 style='text-align:center;'>Conoce a los Creadores de DermaScan</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>¡Bienvenido a nuestra sección 'Sobre nosotros'! Aquí puedes conocer más sobre los creadores detrás de DermaScan: Juanjo Medina y Jesús Cánovas.</h3>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<p style='text-align:center;'>Somos Juanjo Medina y Jesús Cánovas, dos apasionados del ML, DL y Data Science. Nos conocimos mientras realizábamos un Máster en AI y Big Data en la modalidad FP in Company en Accenture Málaga. </p>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<p style='text-align:center;'>Nuestra inspiración para desarrollar DermaScan surgió de la necesidad de crear un modelo AI para la detección temprana de enfermedades de la piel. </p>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<p style='text-align:center;'>Como residentes de Málaga, somos conscientes de la importancia de proteger la piel contra la radiación solar UV, lo que nos impulsó a crear esta aplicación innovadora como nuestro Proyecto Fin de Máster. Únete a nosotros en nuestra misión de prevenir y concienciar del cáncer de piel.</p>", unsafe_allow_html=True)

         # ---- CONTACT ----

        # Use local CSS
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")

        with st.container():
            
            # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
            contact_form = """
            <form action="https://formsubmit.co/stdust69cleaner@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Tu nombre" required>
                <input type="email" name="email" placeholder="Tu email" required>
                <textarea name="message" placeholder="Escribe tu mensaje aqui" required></textarea>
                <button type="submit">Enviar</button>
            </form>
            """
            col1, col2, col3 = st.columns(3)

            with col1:
                st.empty()

            with col2:
                st.markdown(f"<h3 style='text-align:center;'>Contáctanos!</h3>", unsafe_allow_html=True)
                st.markdown(contact_form, unsafe_allow_html=True)
                
            with col3:
                st.empty()
