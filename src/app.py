import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow
import keras
from keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
import os

def main():
    st.title("DermaScan")
    

    # Crear un menú desplegable en la barra lateral
    categoria_seleccionada = st.sidebar.selectbox("Categorías", ["Pagina principal", "Categoría 2", "Categoría 3"])

    # Mostrar la página correspondiente según la categoría seleccionada
    if categoria_seleccionada == "Pagina principal":
        pagina_categoria_1()
    elif categoria_seleccionada == "Categoría 2":
        pagina_categoria_2()
    elif categoria_seleccionada == "Categoría 3":
        pagina_categoria_3()

def pagina_categoria_1():
   
    st.header("Comprueba la salud de tu piel.")
    st.write("Inserta una imagen en el recuadro, que solo salga la piel donde quieras utilizarla.")

    # Agregar un apartado para cargar una foto
    imagen = st.file_uploader("Inserta una imagen", type=["jpg", "jpeg", "png"])

    # Verificar si se cargó una foto
    if imagen is not None:
        st.image(imagen, caption="Imagen cargada", use_column_width=True)
        try:
            clf = keras.models.load_model("model/benigno_vs_maligno_modelo.h5", compile=False)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
        # Convertir la imagen a un formato adecuado para la predicción
        imagen_prueba = Image.open(imagen).convert('RGB')
        imagen_prueba = imagen_prueba.resize((20, 20))# Igualar al modelo original
        imagen_array = np.array(imagen_prueba)
        imagen_array = imagen_array / 255.0  # Normalizar los valores de píxeles entre 0 y 1
        imagen_array = np.expand_dims(imagen_array, axis=0)  # Agregar una dimensión de lote
        imagen_prueba.close()

        try:
            # Realizar la predicción
            prediccion = clf.predict(imagen_array)
            # Imprimir la predicción
            st.write("La prediccion es benigna al : ",prediccion[0, 0])
            st.write("La prediccion es maligna al : ",prediccion[0, 1])
            clase_predicha = np.argmax(prediccion)
            if clase_predicha == 0:
                print("La imagen es benigna.")
            else:
                print("La imagen es maligna.")

        except Exception as e:
            st.error(f"Error al hacer la prediccion: {str(e)}")
        

       

def pagina_categoria_2():
    st.header("Página 2")
    st.write("Contenido pagina 2.")

def pagina_categoria_3():
    st.header("Página 3")
    st.write("Contenido pagina 3.")

if __name__ == "__main__":
    main()
