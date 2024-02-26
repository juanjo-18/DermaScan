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
    st.set_page_config(
        page_title="DermaScan",
        page_icon=":microscope:",
        layout="wide",  # Ancho completo
    )
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
    # Set the layout to two columns
    col1, col2,col3 = st.beta_columns([6,1,3])  # 60% and 40% width
    with col1:
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
                st.write("La prediccion de  piel es: ",prediccion_objeto_piel_modelo[0, 0])
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

   

    # Text in the right column (20%)
    with col3:
        modelo=joblib.load("model/sentimientos_modelo.pkl")  
        def mostrar_imagen_segun_puntuacion(puntuacion):
            if puntuacion == 0:
                st.image("imagenes/estrellas_0.png", caption="", use_column_width=True)
            elif puntuacion == 1:
                st.image("imagenes/estrellas_1.png", caption="", use_column_width=True)
            elif puntuacion == 2:
                st.image("imagenes/estrellas_2.png", caption="", use_column_width=True)
            elif puntuacion == 3:
                st.image("imagenes/estrellas_3.png", caption="", use_column_width=True)
            elif puntuacion == 4:
                st.image("imagenes/estrellas_4.png", caption="", use_column_width=True)
            elif puntuacion == 5:
                st.image("imagenes/estrellas_5.png", caption="", use_column_width=True)

        # Ejemplo de uso
        puntuacion_seleccionada = st.selectbox("Selecciona una puntuación", [0, 1, 2, 3, 4, 5])
        # Casilla de entrada de texto
        #texto_calificacion = st.text_input("Escribe tu comentario:")
        texto_calificacion = st.text_area("Escribe tu comentario:")
        
        # Agrega un botón para borrar el contenido del área de texto
        if st.button("Añadir comentario"):
            texto_calificacion = ""
            st.experimental_rerun()
        
        if len(texto_calificacion.strip()) == 0:
            calificacion=0
        else:
            calificacion=modelo.predict([texto_calificacion])[0]
        
        
        
        st.text(f"¡La puntuacion es de {calificacion}!")
        mostrar_imagen_segun_puntuacion(puntuacion_seleccionada)
    
    

       

def pagina_categoria_2():
    st.header("Página 2")
    

def pagina_categoria_3():
    st.header("Página 3")
    st.write("Contenido pagina 3.")

if __name__ == "__main__":
    main()
