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
import tensorflow as tf

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
            # Ruta al archivo .tflite
            archivo_tflite = 'model/piel_vs_cancer.tflite'

            # Cargar el modelo TensorFlow Lite
            piel_vs_cancer = tf.lite.Interpreter(model_path=archivo_tflite)
            benigno_vs_maligno = keras.models.load_model("model/benigno_vs_maligno_modelo.h5", compile=False)
            clasificador_tipos_cancer = keras.models.load_model("model/clasificador_tipos_cancer.h5", compile=False)
            objeto_piel_modelo = keras.models.load_model("model/objeto_piel_modelo.h5", compile=False)
        
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
        
        # Convertir la imagen a un formato adecuado para la predicción
        imagen = Image.open(imagen).convert('RGB')
        
        # Objeto vs piel
        imagen_objeto_vs_piel = imagen.resize((120, 120))# Igualar al modelo original
        imagen_objeto_vs_piel = np.array(imagen_objeto_vs_piel)
        imagen_objeto_vs_piel = imagen_objeto_vs_piel / 255.0  
        imagen_objeto_vs_piel = np.expand_dims(imagen_objeto_vs_piel, axis=0) 

        # Piel sana y piel cancer
        imagen_piel_sana_vs_cancer = imagen.resize((150, 150))# Igualar al modelo original
        imagen_piel_sana_vs_cancer = np.array(imagen_piel_sana_vs_cancer)
        imagen_piel_sana_vs_cancer = imagen_piel_sana_vs_cancer / 255.0  
        #imagen_piel_sana_vs_cancer = np.expand_dims(imagen_piel_sana_vs_cancer, axis=0) 
        imagen_piel_sana_vs_cancer = np.expand_dims(imagen_piel_sana_vs_cancer.astype(np.float32), axis=0)


        # Benigno vs maligno
        imagen_benigno_vs_maligno = imagen.resize((150, 150))# Igualar al modelo original
        imagen_benigno_vs_maligno = np.array(imagen_benigno_vs_maligno)
        imagen_benigno_vs_maligno = imagen_benigno_vs_maligno / 255.0  
        imagen_benigno_vs_maligno = np.expand_dims(imagen_benigno_vs_maligno, axis=0)  


        # Clasificador tipos
        imagen_clasificador_tipos= imagen.resize((28, 28))# Igualar al modelo original
        imagen_clasificador_tipos = np.array(imagen_clasificador_tipos)
        imagen_clasificador_tipos = imagen_clasificador_tipos / 255.0  
        imagen_clasificador_tipos = np.expand_dims(imagen_clasificador_tipos, axis=0)  
        

        try:
            # Realizar la predicción
            prediccion_objeto_piel_modelo = objeto_piel_modelo.predict(imagen_objeto_vs_piel)
            prediccion_benigno_vs_maligno = benigno_vs_maligno.predict(imagen_benigno_vs_maligno)
            prediccion_clasificador_tipos_cancer = clasificador_tipos_cancer.predict(imagen_clasificador_tipos)
            
            # Asignar memoria para los tensores
            piel_vs_cancer.allocate_tensors()

            entrada_details = piel_vs_cancer.get_input_details()
            salida_details = piel_vs_cancer.get_output_details()
            # Establecer los datos de entrada en el modelo
            
            piel_vs_cancer.set_tensor(entrada_details[0]['index'], imagen_piel_sana_vs_cancer)

            # Ejecutar la inferencia
            piel_vs_cancer.invoke()

            # Obtener los resultados de la inferencia
            resultados = piel_vs_cancer.get_tensor(salida_details[0]['index'])

            # Imprimir la predicción de objeto o piel
            st.write("La prediccion de objeto o piel es: ",prediccion_objeto_piel_modelo[0, 0])
            valor_prediccion_objeto_piel_modelo=prediccion_objeto_piel_modelo[0, 0]
            if valor_prediccion_objeto_piel_modelo >= 0.75:
                st.write("La imagen es piel.")
            else:
                st.write("La imagen es objeto.")
            

            # Imprimir la predicción de piel o piel cancer
            st.write("La prediccion es piel sana al : ",resultados[0][0])
            st.write("La prediccion es piel cancer al : ",resultados[0][1])

            clase_predicha = resultados[0][0]
            if clase_predicha >= 0.75:
                st.write("La imagen es piel sana.")
            else:
                st.write("La imagen es piel cancer.")
            


            # Imprimir la predicción de benigna o maligna
            st.write("La prediccion es benigna al : ",prediccion_benigno_vs_maligno[0, 0])
            st.write("La prediccion es maligna al : ",prediccion_benigno_vs_maligno[0, 1])
            clase_predicha = np.argmax(prediccion_benigno_vs_maligno)
            if clase_predicha == 0:
                st.write("La imagen es benigna.")
            else:
                st.write("La imagen es maligna.")
            
            
           
            
           

        except Exception as e:
            st.error(f"Error al hacer la prediccion: {str(e)}")
        

       

def pagina_categoria_2():
    st.header("Página 2")
    def st_stars(rating, max_rating=5, star_color="yellow"):
        # Calcula el número de estrellas llenas y medias
        full_stars = int(rating)
        half_star = rating - full_stars

        # Crea las estrellas llenas
        stars_html = f'<span style="color: {star_color};">&#9733;</span>' * full_stars

        # Añade una estrella media si es necesario
        if half_star > 0:
            stars_html += f'<span style="color: {star_color};">&#9733;&frac12;</span>'

        # Añade las estrellas vacías necesarias
        empty_stars = max_rating - full_stars - 1
        stars_html += f'<span style="color: {star_color};">&#9734;</span>' * empty_stars

        return stars_html

    # Ejemplo de uso
    puntuacion = st.slider("Selecciona una puntuación", 0.0, 5.0, 3.5, 0.1)
    st.markdown(f"Puntuación: {puntuacion}")
    st.markdown(st_stars(puntuacion))

def pagina_categoria_3():
    st.header("Página 3")
    st.write("Contenido pagina 3.")

if __name__ == "__main__":
    main()
