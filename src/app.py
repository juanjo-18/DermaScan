import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def main():
    st.title("Menú Desplegable de Hamburguesa")

    # Crear un menú desplegable en la barra lateral
    categoria_seleccionada = st.sidebar.selectbox("Categorías", ["Categoría 1", "Categoría 2", "Categoría 3"])

    # Mostrar la página correspondiente según la categoría seleccionada
    if categoria_seleccionada == "Categoría 1":
        pagina_categoria_1()
    elif categoria_seleccionada == "Categoría 2":
        pagina_categoria_2()
    elif categoria_seleccionada == "Categoría 3":
        pagina_categoria_3()

def pagina_categoria_1():
    st.header("Página de la Categoría 1")
    st.write("Contenido específico para la Categoría 1.")

    # Agregar un apartado para cargar una foto
    uploaded_file = st.file_uploader("Inserta una foto", type=["jpg", "jpeg", "png"])

    # Verificar si se cargó una foto
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Foto cargada", use_column_width=True)

def pagina_categoria_2():
    st.header("Página de la Categoría 2")
    st.write("Contenido específico para la Categoría 2.")

def pagina_categoria_3():
    st.header("Página de la Categoría 3")
    st.write("Contenido específico para la Categoría 3.")

if __name__ == "__main__":
    main()
