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


def main():
    st.set_page_config(
        page_title="DermaScan",
        page_icon=":microscope:",
        layout="wide",  # Ancho completo
    )
    st.title("DermaScan")

    # Crear un menú desplegable en la barra lateral
    categoria_seleccionada = st.sidebar.selectbox("MENÚ", ["Inicio", "DermaScan App", "Prevención", "Índice UV", "About us"])

    # Mostrar la página correspondiente según la categoría seleccionada
    if categoria_seleccionada == "Inicio":
        pagina_categoria_1()
    elif categoria_seleccionada == "DermaScan App":
        pagina_categoria_2()
    elif categoria_seleccionada == "Prevención":
        pagina_categoria_3()
    elif categoria_seleccionada == "Índice UV":
        pagina_categoria_4()
    elif categoria_seleccionada == "About us":
        pagina_categoria_5()



def pagina_categoria_1():
# Encabezado principal
    st.header("¡Bienvenidos a DermaScan!")
    st.subheader("Descubre el Futuro del Cuidado de la Piel")

    col1, col2 = st.columns([1, 1])
    with col1:
    # Sección: DermaScan App
        st.subheader("DermaScan App: Tu Aliado en la Lucha Contra el Cáncer de Piel")
        st.write("DermaScan App representa la vanguardia en el cuidado de la piel, utilizando modelos de inteligencia artificial para analizar imágenes y detectar posibles lesiones cutáneas. Nuestra aplicación puede identificar si una lesión es maligna, así como clasificarla según su tipo, proporcionando a los usuarios una herramienta poderosa para la detección temprana y la prevención del cáncer de piel.")
    
    with col2:
    # Sección: Prevención y cuidado de la piel
        st.subheader("Prevención y Cuidado de la Piel: Tu Guía hacia una Piel Saludable")
        st.write("En esta sección, encontrarás una amplia gama de recomendaciones y consejos sobre cómo mantener tu piel saludable y protegida contra los daños solares. Desde prácticas diarias de cuidado de la piel hasta medidas preventivas contra el cáncer de piel, estamos aquí para ayudarte a mantener una piel radiante y saludable en todo momento.")

    col3, col4 = st.columns([1, 1])
    with col3:
    # Sección: Incidencia Solar UV
        st.subheader("Incidencia Solar UV: Conoce el Impacto del Sol en tu Piel")
        st.write("Descubre la importancia de estar conscienciado del índice UV y su impacto en tu piel. En esta sección, te proporcionamos información en tiempo real sobre el índice UV actual en diferentes regiones, así como pronósticos para los próximos días. Mantente informado y protege tu piel contra los daños causados por la radiación solar.")

    with col4:
    # Sección: About us
        st.subheader("About us: Conoce a los Creadores de DermaScan")
        st.write("Somos Juanjo Medina y Jesús Cánovas, dos apasionados del aprendizaje automático y la ciencia de datos. Nuestra inspiración para desarrollar DermaScan surgió de la necesidad de crear un modelo de inteligencia artificial para la detección temprana de enfermedades de la piel. Como residentes de Málaga, somos conscientes de la importancia de proteger la piel contra la radiación solar UV, lo que nos impulsó a crear esta aplicación innovadora como nuestro proyecto final de máster. Únete a nosotros en nuestra misión sobre el concienciamiento y la precvención del cáncer de piel.")



def pagina_categoria_2():
    calificacion = 0
    # Set the layout to two columns
    col5, col6,col7 = st.columns([6,1,3])  
    with col5:
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

   
    # COLUMNA DE RATINGS (20%)
    with col7:
        # Llamar a la función para mostrar los datos desde S3
        def guardar_puntuacion_en_s3(comentario, puntuacion):
            # Convierte el comentario y la puntuación en un DataFrame
            df_nuevo = pd.DataFrame({"COMENTARIO": [comentario], "PUNTUACION": [puntuacion]})

            try:
                # Conecta con S3 y lee el archivo existente
                s3 = boto3.client('s3', aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"], aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
                obj = s3.get_object(Bucket='dermascan-streamlits3', Key='pruebas3_streamlit.csv')
                df_existente = pd.read_csv(obj['Body'])

                # Concatena el nuevo DataFrame con el existente
                df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)

                # Escribe el DataFrame final en S3
                s3.put_object(Bucket='dermascan-streamlits3', Key='pruebas3_streamlit.csv', Body=df_final.to_csv(index=False), ContentType='text/csv')
            except NoCredentialsError:
                st.error("No se encontraron las credenciales de AWS. Por favor, configure sus credenciales correctamente.")

        
        def mostrar_datos_desde_s3():
            try:
                # Conecta con S3 y lee el archivo CSV
                s3 = boto3.client('s3', aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"], aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
                obj = s3.get_object(Bucket='dermascan-streamlits3', Key='pruebas3_streamlit.csv')
                df = pd.read_csv(obj['Body'])

                suma_puntuaciones = df['PUNTUACION'].sum()
                # Calcula la cantidad de registros en la columna 'puntuaciones'
                cantidad_registros = len(df)

                # Calcula el promedio dividiendo la suma total por la cantidad de registros
                promedio = suma_puntuaciones / cantidad_registros
                mostrar_imagen_segun_puntuacion(int(promedio))
                st.markdown(f"<p style='text-align:center;'>¡La puntuación media es de {round(promedio,2)}!</p>", unsafe_allow_html=True)
              
                # Muestra el DataFrame en Streamlit
                #st.write(df)

                # Ordenar el DataFrame por puntuación en orden descendente
                df_ordenado = df.sort_values(by='PUNTUACION', ascending=False)

                # Extraer los tres mejores textos y asignarlos a variables separadas
                comentario1 = df_ordenado.iloc[0]['COMENTARIO']
                puntuacion1= round(df_ordenado.iloc[0]['PUNTUACION'],2)
                comentario2 = df_ordenado.iloc[1]['COMENTARIO']
                puntuacion2= round(df_ordenado.iloc[1]['PUNTUACION'],2)
                comentario3 = df_ordenado.iloc[2]['COMENTARIO']
                puntuacion3= round(df_ordenado.iloc[2]['PUNTUACION'],2)
                
                mejor_texto1 = f'1. "{comentario1}"  -  {puntuacion1}'
                mejor_texto2 = f'2. "{comentario2}"  -  {puntuacion2}'
                mejor_texto3 = f'3. "{comentario3}"  -  {puntuacion3}'

                # Ordenar el DataFrame por puntuación en orden ascendente
                df_ordenado = df.sort_values(by='PUNTUACION', ascending=True)

                # Extraer los tres peores textos y asignarlos a variables separadas
                comentario1 = df_ordenado.iloc[0]['COMENTARIO']
                puntuacion1= round(df_ordenado.iloc[0]['PUNTUACION'],2)
                comentario2 = df_ordenado.iloc[1]['COMENTARIO']
                puntuacion2= round(df_ordenado.iloc[1]['PUNTUACION'],2)
                comentario3 = df_ordenado.iloc[2]['COMENTARIO']
                puntuacion3= round(df_ordenado.iloc[2]['PUNTUACION'],2)
                
                peor_texto1 = f'1. "{comentario1}"  -  {puntuacion1}'
                peor_texto2 = f'2. "{comentario2}"  -  {puntuacion2}'
                peor_texto3 = f'3. "{comentario3}"  -  {puntuacion3}'

                # Crear un bloque vacío de 10px de altura usando HTML personalizado
                bloque_vacio = '<div style="height: 20px;"></div>'
                st.markdown(bloque_vacio, unsafe_allow_html=True)
                crear_bloques_reseñas(mejor_texto1,mejor_texto2,mejor_texto3,"Las 3 mejores reseñas:")
                st.markdown(bloque_vacio, unsafe_allow_html=True)
                crear_bloques_reseñas(peor_texto1,peor_texto2,peor_texto3,"Las 3 peores reseñas:")
                
            except NoCredentialsError:
                st.error("No se encontraron las credenciales de AWS. Por favor, configure sus credenciales correctamente.")
        

        def crear_bloques_reseñas(texto,texto1,texto2,frase):
            # Concatenar los textos
            textos_concatenados = f"{texto}<br>{texto1}<br>{texto2}"

            estilo_bloque = (
            "background-color: white; "
            "color: black; "
            "padding: 10px; "
            "border-radius: 5px; "
            "box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);"
            )
            # Mostrar el bloque en blanco con el texto encima
            st.write(frase)
            with st.container():
                st.markdown('<div style="{}">{}</div>'.format(estilo_bloque, textos_concatenados), unsafe_allow_html=True)


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
        
        st.header("Ratings")
        # Casilla de entrada de texto
        st.write("Púntua nuestra pagina dejando un comentario: ")
        # text_area para ingresar el comentario
        texto_calificacion = st.text_input("")
        
        # Centra el botón utilizando st.button y estilo CSS
        button_html = """
            <style>
                div.stButton > button {
                    width: 100%;
                    display: flex;
                    justify-content: center;
                }
            </style>
        """
        st.markdown(button_html, unsafe_allow_html=True)

        # Agrega un botón para borrar el contenido del área de texto
        if st.button("Enviar"):
             # Llamar a la función para guardar los datos en S3
            if len(texto_calificacion.strip()) > 0:
                calificacion=modelo.predict([texto_calificacion])[0]
                st.write("Tu calificacion a sido de: ",calificacion)
                guardar_puntuacion_en_s3(texto_calificacion, calificacion)
        mostrar_datos_desde_s3()
        
        

def pagina_categoria_3():
    st.header("Prevención y cuidado de la piel")
    
    st.subheader("Recomendaciones Generales")
    
    # División de la pantalla en columnas proporcionales
    col8, col9, col10 = st.columns([1, 1, 1])
    
    # Sección de Tiempo de Exposición
    with col8:
        st.subheader("Tiempo de Exposición al Sol")
        st.image("imagenes/sun_1.jpg", use_column_width=True) 
        st.write("Limita tu tiempo de exposición al sol, especialmente durante las horas pico de radiación solar.")
        st.write("Usa ropa protectora, sombreros de ala ancha y busca sombra para reducir la exposición directa al sol.")
        st.write("Aplica protector solar de amplio espectro con factor de protección solar (FPS) 30 o superior cada dos horas, o más frecuentemente si sudas o te mojas.")

    # Sección de Hidratación
    with col9:
        st.subheader("Hidratación")
        st.image("imagenes/water_1.jpg", use_column_width=True)
        st.write("Mantente bien hidratado bebiendo suficiente agua a lo largo del día, especialmente en climas cálidos o cuando haces ejercicio.")
        st.write("Considera el consumo de frutas y verduras con alto contenido de agua, como sandía, pepino y naranjas, para ayudar a mantener la hidratación.")
        st.write("Evita el exceso de cafeína y alcohol, ya que pueden tener un efecto deshidratante en el cuerpo.")

    # Sección de Dieta Saludable
    with col10:
        st.subheader("Dieta Saludable")
        st.image("imagenes/food_1.jpg", use_column_width=True)
        st.write("Prioriza una dieta rica en frutas, verduras, granos enteros y proteínas magras para obtener los nutrientes necesarios para una piel saludable.")
        st.write("Los alimentos ricos en antioxidantes, como las bayas, el té verde y las nueces, pueden ayudar a proteger la piel del daño causado por los radicales libres.")
        st.write("Evita los alimentos procesados, altos en azúcares añadidos y grasas saturadas, ya que pueden contribuir a problemas cutáneos como el acné y la inflamación.")
    
    # División de la pantalla en columnas proporcionales
    col11, col12, col13 = st.columns([1, 1, 1])

    # Sección de Protección Solar
    with col11:
        st.subheader("Protección Solar")
        st.image("imagenes/sun_2.jpg", use_column_width=True)
        st.write("Usa protector solar incluso en días nublados o cuando estés bajo la sombra, ya que los rayos UV pueden penetrar las nubes y reflejarse en superficies como la arena y el agua.")
        st.write("Elige un protector solar que ofrezca protección de amplio espectro contra los rayos UVA y UVB, y asegúrate de aplicarlo generosamente en todas las áreas expuestas de la piel.")
        st.write("No te olvides de proteger tus labios con un bálsamo labial con SPF y usar gafas de sol con protección UV para proteger tus ojos del daño solar.")
        

    # Sección de otra recomendación
    with col12:
        st.subheader("Visita al dermatólogo")
        st.image("imagenes/zderm_1.jpg", use_column_width=True)
        st.write("Programa revisiones periódicas con un dermatólogo para realizar un seguimiento de la salud de tu piel y detectar cualquier cambio o problema potencial de manera temprana.")
        st.write("El dermatólogo puede ofrecerte consejos personalizados sobre cómo cuidar y proteger tu piel, así como recomendaciones específicas de productos y tratamientos adecuados para tus necesidades individuales.")
        st.write("No subestimes la importancia de consultar regularmente a un dermatólogo, especialmente si tienes antecedentes de problemas cutáneos, exposición solar frecuente o cambios en la apariencia de lunares o manchas en la piel.")

    # Sección de otra recomendación
    with col13:
        st.subheader("Estilo de vida saludable")
        st.image("imagenes/sport_1.jpg", use_column_width=True)
        st.write("Adopta un estilo de vida saludable que incluya hábitos como una alimentación equilibrada, ejercicio regular y descanso adecuado para promover la salud general de tu piel.")
        st.write("Evita el tabaquismo y reduce el consumo de alcohol, ya que estos hábitos pueden afectar negativamente la salud de la piel y aumentar el riesgo de problemas cutáneos como el envejecimiento prematuro y el cáncer de piel.")
        st.write("Mantén un equilibrio entre el trabajo y el descanso, y encuentra formas de gestionar el estrés, ya que el estrés crónico puede contribuir a problemas cutáneos como el acné y la dermatitis.")



def pagina_categoria_4():
    st.header("Índice UV")
    st.write("Contenido pagina 3.")

def pagina_categoria_5():
    st.header("Página 4")
    st.write("Contenido pagina 4.")

if __name__ == "__main__":
    main()