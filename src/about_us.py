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
        st.markdown(f"<h2 style='text-align:center;'>¡Bienvenido a nuestra sección 'Sobre nosotros'! Aquí puedes conocer más sobre los creadores detrás de DermaScan: Juanjo Medina y Jesús Cánovas.</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Somos Juanjo Medina y Jesús Cánovas, dos apasionados del aprendizaje automático y la ciencia de datos. Nos conocimos mientras realizábamos un Máster en inteligencia artificial y Big Data en el centro integrado de formación profesional Alan Turing, y desde entonces hemos estado trabajando juntos en proyectos innovadores. La idea de desarrollar DermaScan surgió de nuestra pasión por la tecnología y nuestra conciencia sobre los desafíos en la detección temprana del cáncer de piel. Únete a nosotros en nuestra misión de concienciar y prevenir el cáncer de piel mientras exploramos los límites de la inteligencia artificial y la salud digital.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Nuestra inspiración para desarrollar DermaScan surgió de la necesidad de crear un modelo de inteligencia artificial para la detección temprana de enfermedades de la piel. </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Como residentes de Málaga, somos conscientes de la importancia de proteger la piel contra la radiación solar UV, lo que nos impulsó a crear esta aplicación innovadora como nuestro proyecto final de máster. Únete a nosotros en nuestra misión sobre el concienciamiento y la precvención del cáncer de piel.</p>", unsafe_allow_html=True)

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
                st.header("Contáctanos!")
                st.markdown(contact_form, unsafe_allow_html=True)
            with col3:
                st.empty()
