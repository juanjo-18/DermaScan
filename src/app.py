import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

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
    uploaded_file = st.file_uploader("Inserta una imagen", type=["jpg", "jpeg", "png"])

    # Verificar si se cargó una foto
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

def pagina_categoria_2():
    st.header("Página 2")
    st.write("Contenido pagina 2.")

def pagina_categoria_3():
    st.header("Página 3")
    st.write("Contenido pagina 3.")

if __name__ == "__main__":
    main()
