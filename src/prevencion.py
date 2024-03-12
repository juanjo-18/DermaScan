import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import tensorflow
import keras
import boto3
import os
import spacy

from keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
from st_files_connection import FilesConnection
from botocore.exceptions import NoCredentialsError

from hydralit import HydraApp
from hydralit import HydraHeadApp



# CREMOS UNA CLASE PARA LA PAGINA
class Prevencion(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):

        @st.cache(allow_output_mutation=True)
        def get_base64_of_bin_file(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()

        def set_png_as_page_bg(png_file):
            bin_str = get_base64_of_bin_file(png_file)
            page_bg_img = '''
            <style>
            body {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
            }
            </style>
            ''' % bin_str
            
            st.markdown(page_bg_img, unsafe_allow_html=True)
            return

        set_png_as_page_bg('background.png')
        
        st.markdown(f"<h1 style='text-align:center;'> Cuidado y Prevención: Tu piel siempre saludable </h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'> ¡Bienvenido a nuestra sección de Prevención y Cuidado de la Piel! Aquí encontrarás una completa guía para mantener tu piel saludable y protegida en todo momento. Desde consejos y recomendaciones para protegerte del sol y mantener una piel saludable. Descubre cómo adoptar prácticas de cuidado de la piel que promuevan la salud a largo plazo.</p>", unsafe_allow_html=True)
        st.divider()

        col1, col2, col3, col4, col5, col6, col7= st.columns([0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 0.5])

        with col2:
        # Sección: DermaScan App
            st.markdown(f"<h3 style='text-align:center;'> Tiempo de Exposición al Sol </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Limita tu tiempo de exposición al sol, especialmente durante las horas pico de radiación solar.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Usa ropa protectora, sombreros de ala ancha y busca sombra para reducir la exposición directa al sol.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Aplica protector solar de amplio espectro con factor de protección solar (FPS) 30 o superior cada dos horas, o más frecuentemente si sudas o te mojas.</p>", unsafe_allow_html=True)

            style_image1 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/sun1_prev.jpg" height="600" style="{style_image1}">', unsafe_allow_html=True)
            
    
        with col4:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Hidratación </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Mantente bien hidratado bebiendo suficiente agua a lo largo del día, especialmente en climas cálidos o cuando haces ejercicio.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Considera el consumo de frutas y verduras con alto contenido de agua, como sandía, pepino y naranjas, para ayudar a mantener la hidratación.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Evita el exceso de cafeína y alcohol, ya que pueden tener un efecto deshidratante en el cuerpo.</p>", unsafe_allow_html=True)
            
            style_image2 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/agua2_prev.jpg" height="600" style="{style_image2}">', unsafe_allow_html=True)
            

        with col6:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Dieta Saludable </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>Prioriza una dieta rica en frutas, verduras, granos enteros y proteínas magras para obtener los nutrientes necesarios para una piel saludable.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Los alimentos ricos en antioxidantes, como las bayas, el té verde y las nueces, pueden ayudar a proteger la piel del daño causado por los radicales libres.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Evita los alimentos procesados, altos en azúcares añadidos y grasas saturadas, ya que pueden contribuir a problemas cutáneos como el acné y la inflamación.</p>", unsafe_allow_html=True)

            style_image3 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/food3_prev.jpg" height="600" style="{style_image3}">', unsafe_allow_html=True)
            
        st.divider()

        col8, col9, col10, col11, col12, col13, col14= st.columns([0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 0.5])

        with col9:
        # Sección: DermaScan App
            st.markdown(f"<h3 style='text-align:center;'> Protección Solar </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Usa protector solar incluso en días nublados o cuando estés bajo la sombra, ya que los rayos UV pueden penetrar las nubes y reflejarse en superficies como la arena y el agua.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Elige un protector solar que ofrezca protección de amplio espectro contra los rayos UVA y UVB, y asegúrate de aplicarlo generosamente en todas las áreas expuestas de la piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>No te olvides de proteger tus labios con un bálsamo labial con SPF y usar gafas de sol con protección UV para proteger tus ojos del daño solar.</p>", unsafe_allow_html=True)

            style_image4 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/protec4_prev.jpg" height="600" style="{style_image4}">', unsafe_allow_html=True)

    
        with col11:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Visita al dermatólogo </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Programa revisiones periódicas con un dermatólogo para realizar un seguimiento de la salud de tu piel y detectar cualquier cambio o problema potencial de manera temprana.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>El dermatólogo puede ofrecerte consejos personalizados sobre cómo cuidar y proteger tu piel, así como recomendaciones específicas de productos y tratamientos adecuados para tus necesidades individuales.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>No subestimes la importancia de consultar regularmente a un dermatólogo, especialmente si tienes antecedentes de problemas cutáneos, exposición solar frecuente o cambios en la apariencia de lunares o manchas en la piel.</p>", unsafe_allow_html=True)
            
            style_image5 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/derm5_prev.jpg" height="600" style="{style_image5}">', unsafe_allow_html=True)
            

        with col13:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Estilo de vida saludable </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>Adopta un estilo de vida saludable que incluya hábitos como una alimentación equilibrada, ejercicio regular y descanso adecuado para promover la salud general de tu piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>vita el tabaquismo y reduce el consumo de alcohol, ya que estos hábitos pueden afectar negativamente la salud de la piel y aumentar el riesgo de problemas cutáneos como el envejecimiento prematuro y el cáncer de piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Mantén un equilibrio entre el trabajo y el descanso, y encuentra formas de gestionar el estrés, ya que el estrés crónico puede contribuir a problemas cutáneos como el acné y la dermatitis.</p>", unsafe_allow_html=True)

            style_image6 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/sport6_prev.jpg" height="600" style="{style_image6}">', unsafe_allow_html=True)

        st.divider()

        # CHATBOT: DermIA
        # Cargar modelo de lenguaje en español
        nlp = spacy.load("es_core_news_sm")

        # Función para calcular la similitud entre dos textos
        def calcular_similitud(texto1, texto2):
            doc1 = nlp(texto1)
            doc2 = nlp(texto2)
            return doc1.similarity(doc2)

        # Función para responder preguntas
        def responder_pregunta(pregunta):
            respuesta = ""
            max_similitud = 0
            
            pregunta = pregunta.lower()  # Convertir la pregunta a minúsculas para facilitar la comparación

            # Lista de preguntas frecuentes y respuestas asociadas
            preguntas_frecuentes = {
                "tiempo de exposición al sol": "Para protegerte del sol, es importante limitar tu tiempo de exposición, especialmente durante las horas pico de radiación solar. Recuerda usar ropa protectora, sombreros de ala ancha y buscar sombra para reducir la exposición directa.",
                "hidratación": "Mantente bien hidratado bebiendo suficiente agua a lo largo del día, especialmente en climas cálidos o cuando haces ejercicio. También puedes consumir frutas y verduras con alto contenido de agua, como sandía, pepino y naranjas.",
                "dieta saludable": "Prioriza una dieta rica en frutas, verduras, granos enteros y proteínas magras para obtener los nutrientes necesarios para una piel saludable. Los alimentos ricos en antioxidantes, como las bayas y el té verde, pueden ayudar a proteger la piel del daño.",
                "protección solar": "Es importante usar protector solar incluso en días nublados o bajo la sombra, ya que los rayos UV pueden penetrar las nubes y reflejarse en superficies como la arena y el agua. Recuerda aplicarlo generosamente en todas las áreas expuestas de la piel.",
                "visita al dermatólogo": "Programa revisiones periódicas con un dermatólogo para realizar un seguimiento de la salud de tu piel y detectar cualquier cambio o problema potencial de manera temprana. El dermatólogo puede ofrecerte consejos personalizados sobre cuidado y protección de la piel.",
                "estilo de vida saludable": "Adopta un estilo de vida saludable que incluya hábitos como una alimentación equilibrada, ejercicio regular y descanso adecuado para promover la salud general de tu piel. Evita el tabaquismo y reduce el consumo de alcohol.",
                "¿Cómo puedo protegerme del sol?": "Puedes protegerte del sol limitando tu tiempo de exposición, usando ropa protectora, sombreros y aplicando protector solar de amplio espectro.",
                "¿Con qué frecuencia debo aplicar protector solar?": "Debes aplicar protector solar cada dos horas, o más frecuentemente si sudas o te mojas.",
                "¿Qué debo hacer para mantenerme hidratado?": "Mantente bien hidratado bebiendo suficiente agua a lo largo del día y considera consumir frutas y verduras con alto contenido de agua.",
                "¿Qué tipo de dieta es buena para mi piel?": "Una dieta rica en frutas, verduras, granos enteros y proteínas magras es buena para la salud de la piel.",
                "¿Por qué es importante usar protector solar incluso en días nublados?": "Es importante usar protector solar incluso en días nublados porque los rayos UV pueden penetrar las nubes y causar daño a la piel.",
                "¿Qué debo esperar en una visita al dermatólogo?": "En una visita al dermatólogo, puedes esperar revisiones periódicas de la salud de tu piel y recibir consejos personalizados sobre cuidado y protección de la piel.",
                "¿Cómo puedo adoptar un estilo de vida saludable para mi piel?": "Puedes adoptar un estilo de vida saludable para tu piel incluyendo hábitos como una alimentación equilibrada, ejercicio regular y descanso adecuado.",
                "¿Cómo puedo reducir el estrés para mejorar la salud de mi piel?": "Puedes reducir el estrés practicando técnicas de gestión del estrés como la meditación, el yoga o el ejercicio regular.",
                "¿Qué debo evitar para proteger mi piel?": "Debes evitar el tabaquismo, el consumo excesivo de alcohol y la exposición prolongada al sol sin protección.",
                "¿Qué debo hacer si noto cambios en mi piel?": "Debes programar una cita con un dermatólogo para evaluar cualquier cambio en tu piel y recibir el tratamiento adecuado si es necesario.",
                "¿Cuál es la mejor hora del día para evitar la exposición al sol?": "La mejor hora para evitar la exposición al sol es entre las 10 a.m. y las 4 p.m., cuando los rayos solares son más intensos.",
                "¿Qué debo hacer si me quemo con el sol?": "Si te quemas con el sol, debes aplicar compresas frías sobre la piel afectada, hidratarte abundantemente y evitar exponerte al sol hasta que la quemadura sane.",
                "¿Qué debo hacer si tengo la piel seca?": "Si tienes la piel seca, debes usar cremas hidratantes regularmente, evitar baños o duchas muy calientes y beber suficiente agua para mantener la piel hidratada desde adentro.",
                "¿Cuál es el mejor tipo de protector solar para mi piel?": "El mejor tipo de protector solar depende de tu tipo de piel y preferencias personales. Busca uno que ofrezca protección de amplio espectro y tenga un FPS adecuado para tu tipo de piel.",
                "¿Es necesario usar protector solar en días nublados?": "Sí, es necesario usar protector solar incluso en días nublados, ya que los rayos UV pueden penetrar las nubes y causar daño a la piel.",
                "¿Cómo puedo prevenir el envejecimiento prematuro de la piel?": "Puedes prevenir el envejecimiento prematuro de la piel usando protector solar diariamente, evitando fumar, manteniendo una dieta saludable y usando productos para el cuidado de la piel que contengan antioxidantes.",
                "¿Qué debo hacer si tengo antecedentes familiares de cáncer de piel?": "Si tienes antecedentes familiares de cáncer de piel, debes ser especialmente diligente en la protección solar, evitar la exposición excesiva al sol y realizar exámenes regulares de la piel con un dermatólogo.",
                "¿Es importante proteger los labios del sol?": "Sí, es importante proteger los labios del sol usando bálsamo labial con SPF, ya que los labios también pueden quemarse y desarrollar cáncer de piel.",
                "¿Qué debo hacer si mi piel está irritada por el sol?": "Si tu piel está irritada por el sol, debes aplicar compresas frías, evitar la exposición adicional al sol y usar lociones calmantes como aloe vera para ayudar a aliviar la irritación.",
                "¿Cuánto tiempo debo esperar después de aplicar protector solar antes de salir al sol?": "Debes esperar aproximadamente 15 a 30 minutos después de aplicar protector solar antes de salir al sol para permitir que se absorba en la piel y proporcione protección efectiva.",
                "¿Qué significa SPF en los protectores solares?": "SPF significa Factor de Protección Solar y representa la capacidad de un protector solar para proteger la piel contra los rayos UVB.",
                "¿Qué debo buscar al elegir un protector solar?": "Debes buscar un protector solar que ofrezca protección de amplio espectro contra los rayos UVA y UVB, y tenga un FPS de al menos 30.",
                "¿Puedo usar protector solar en mi cara todos los días?": "Sí, es recomendable usar protector solar en la cara todos los días, incluso en días nublados, para protegerla del daño solar.",
                "¿Cómo puedo proteger mis ojos del sol?": "Puedes proteger tus ojos del sol usando gafas de sol con protección UV cuando estés al aire libre, especialmente durante las horas pico de radiación solar.",
                "¿El protector solar es seguro para todos los tipos de piel?": "Sí, el protector solar es seguro para todos los tipos de piel, incluyendo piel sensible y propensa al acné. Solo asegúrate de elegir un protector solar adecuado para tu tipo de piel.",
                "¿Cuándo debo comenzar a aplicar protector solar en los niños?": "Debes comenzar a aplicar protector solar en los niños a partir de los 6 meses de edad. Utiliza un protector solar especialmente formulado para bebés y niños pequeños.",
                "¿Qué debo hacer si me quemo con el sol?": "Si te quemas con el sol, enfría la piel con compresas frías o una ducha fría, y aplica loción hidratante o aloe vera para aliviar la irritación.",
                "¿El bronceado en interiores es seguro?": "No, el bronceado en interiores no es seguro y puede aumentar el riesgo de cáncer de piel y envejecimiento prematuro de la piel. Es mejor evitarlo y optar por broncearse de forma natural con protección solar adecuada.",
                "¿Cómo puedo mantener mi piel hidratada en climas secos?": "Para mantener tu piel hidratada en climas secos, utiliza cremas hidratantes y humectantes regularmente y bebe suficiente agua a lo largo del día.",
                "¿Los productos de cuidado de la piel caducan?": "Sí, los productos de cuidado de la piel tienen una fecha de vencimiento y pueden perder efectividad con el tiempo. Verifica las fechas de vencimiento en los envases y desecha los productos vencidos.",
                "¿Qué es el cáncer de piel y cómo puedo prevenirlo?": "El cáncer de piel es un crecimiento anormal de células en la piel que puede ser causado por la exposición al sol. Puedes prevenir el cáncer de piel protegiéndote del sol, usando protector solar y realizando exámenes regulares de la piel.",
                "¿Qué es la vitamina D y por qué es importante para la piel?": "La vitamina D es una vitamina soluble en grasa que es importante para la salud ósea y el sistema inmunológico. La piel produce vitamina D cuando se expone al sol, pero es importante equilibrar la exposición al sol para evitar el daño solar.",
                "¿Cuál es la mejor manera de cuidar la piel después de tomar sol?": "Después de tomar sol, es importante hidratar la piel con loción o crema hidratante para ayudar a calmar la piel y prevenir la sequedad. Evita la exposición adicional al sol y usa ropa protectora si es necesario.",
                "¿Puedo usar protector solar en mi cabello?": "Sí, puedes usar protector solar en tu cabello para protegerlo del daño solar y la decoloración. Busca productos capilares con protección UV o usa un sombrero para proteger tu cuero cabelludo y cabello del sol.",
                "¿Qué debo hacer si tengo una quemadura solar grave?": "Si tienes una quemadura solar grave, busca atención médica si experimentas ampollas, fiebre o dolor intenso. Enfría la piel con compresas frías y bebe mucha agua para prevenir la deshidratación.",
                "¿Qué tipo de ropa debo usar para protegerme del sol?": "Debes usar ropa de manga larga, sombreros de ala ancha y gafas de sol con protección UV para protegerte del sol. Busca ropa con factor de protección solar (FPS) incorporado para una protección adicional.",
                "¿Qué debo hacer si tengo una alergia al sol?": "Si tienes una alergia al sol, evita la exposición directa al sol y usa protector solar y ropa protectora. Consulta a un dermatólogo para obtener tratamiento y consejos sobre cómo manejar la alergia al sol.",
                "¿Qué es la exfoliación y cómo beneficia a la piel?": "La exfoliación es el proceso de eliminar las células muertas de la piel para revelar una piel más suave y radiante debajo. La exfoliación regular puede ayudar a mejorar la textura de la piel y prevenir los brotes de acné.",
                "¿El protector solar es seguro para usar durante el embarazo?": "Sí, el protector solar es seguro para usar durante el embarazo y es importante protegerse del sol para prevenir el melasma y otras afecciones de la piel asociadas con el embarazo.",
                "¿Qué puedo hacer para reducir las manchas oscuras en la piel?": "Para reducir las manchas oscuras en la piel, usa protector solar diariamente, evita la exposición al sol y considera el uso de productos despigmentantes o tratamientos dermatológicos según las recomendaciones de un profesional.",
                "¿Cuáles son los signos y síntomas del cáncer de piel?": "Los signos y síntomas del cáncer de piel pueden incluir cambios en la forma, color o tamaño de lunares o manchas en la piel, así como la aparición de nuevas lesiones que no sanan o que sangran.",
                "¿Qué debo hacer si encuentro un lunar sospechoso en mi piel?": "Si encuentras un lunar sospechoso en tu piel, consulta a un dermatólogo para una evaluación completa. El médico puede realizar una biopsia para determinar si la lesión es cancerosa y ofrecerte el tratamiento adecuado si es necesario.",
                "¿Qué factores aumentan el riesgo de desarrollar cáncer de piel?": "Los factores que aumentan el riesgo de desarrollar cáncer de piel incluyen la exposición prolongada al sol, antecedentes familiares de cáncer de piel, piel clara y antecedentes de quemaduras solares graves.",
                "¿Es importante protegerse del sol incluso en climas fríos?": "Sí, es importante protegerse del sol incluso en climas fríos, ya que los rayos UV pueden penetrar las nubes y causar daño a la piel. Usa protector solar y ropa protectora incluso en días nublados o fríos.",
                "¿Qué es el melanoma y cómo puedo prevenirlo?": "El melanoma es un tipo de cáncer de piel que puede ser mortal si no se detecta y trata a tiempo. Puedes prevenir el melanoma protegiéndote del sol, evitando el bronceado en interiores y realizando autoexámenes regulares de la piel.",
                "¿El bronceado en interiores aumenta el riesgo de cáncer de piel?": "Sí, el bronceado en interiores aumenta el riesgo de cáncer de piel, incluido el melanoma. Es mejor evitar el bronceado en interiores y optar por broncearse de forma natural con protección solar adecuada.",
                "¿Qué debo hacer si tengo una quemadura solar grave?": "Si tienes una quemadura solar grave, busca atención médica si experimentas ampollas, fiebre o dolor intenso. Enfría la piel con compresas frías y bebe mucha agua para prevenir la deshidratación.",
                "¿Cuáles son los efectos a largo plazo de la exposición al sol sin protección?": "La exposición prolongada al sol sin protección puede aumentar el riesgo de cáncer de piel, envejecimiento prematuro de la piel, arrugas, manchas oscuras y otros problemas cutáneos.",
                "¿Puedo tener cáncer de piel si no me quemo con el sol?": "Sí, es posible desarrollar cáncer de piel incluso sin quemarse con el sol. La exposición prolongada al sol sin protección puede aumentar el riesgo de cáncer de piel, incluso si no experimentas quemaduras solares.",
                "¿Qué debo hacer si tengo antecedentes familiares de cáncer de piel?": "Si tienes antecedentes familiares de cáncer de piel, es importante protegerse del sol y realizar exámenes regulares de la piel. Consulta a un dermatólogo para recibir consejos personalizados sobre cómo cuidar y proteger tu piel.",
                "¿Qué es la queratosis actínica y cómo se trata?": "La queratosis actínica es una lesión cutánea precancerosa causada por la exposición al sol. Se puede tratar con crioterapia, terapia fotodinámica o extirpación quirúrgica.",
                "¿Cuál es la diferencia entre el carcinoma de células basales y el carcinoma de células escamosas?": "El carcinoma de células basales y el carcinoma de células escamosas son dos tipos comunes de cáncer de piel no melanoma. El carcinoma de células basales suele aparecer como una protuberancia perlada o una llaga que no sana, mientras que el carcinoma de células escamosas puede manifestarse como una lesión roja y escamosa o una úlcera que no cicatriza.",
                "¿Qué es el eritema solar y cómo se trata?": "El eritema solar es la respuesta inflamatoria de la piel a la exposición excesiva al sol. Se manifiesta como enrojecimiento y sensibilidad en la piel. Se puede tratar con compresas frías, lociones calmantes y analgésicos de venta libre.",
                "¿Es cierto que el sol puede ayudar a tratar el acné?": "Si bien la exposición moderada al sol puede secar temporalmente las lesiones de acné y reducir la inflamación, la exposición excesiva al sol puede empeorar el acné a largo plazo y aumentar el riesgo de daño solar.",
                "¿Puede el sol empeorar las condiciones de la piel como el eczema o la psoriasis?": "Sí, la exposición al sol puede empeorar las condiciones de la piel como el eczema y la psoriasis en algunas personas. Es importante proteger la piel y evitar la exposición prolongada al sol si tienes una condición de la piel sensible.",
                "¿Qué es el lentigo solar y cómo se trata?": "El lentigo solar, también conocido como mancha de la edad o mancha solar, es una lesión pigmentada causada por la exposición crónica al sol. Se puede tratar con crioterapia, terapia láser o extirpación quirúrgica.",
                "¿El sol puede causar cáncer de piel en personas con piel oscura?": "Sí, todas las personas, independientemente de su color de piel, pueden desarrollar cáncer de piel debido a la exposición al sol. Aunque las personas con piel más oscura tienen un menor riesgo, aún pueden desarrollar cáncer de piel, especialmente en áreas menos pigmentadas como las palmas de las manos y las plantas de los pies.",
                "¿El uso de camas de bronceado es seguro?": "No, el uso de camas de bronceado no es seguro y aumenta el riesgo de cáncer de piel, incluido el melanoma. Las camas de bronceado emiten radiación ultravioleta que puede causar daño a la piel y aumentar el riesgo de cáncer.",
                "¿Qué es el fotoenvejecimiento y cómo puedo prevenirlo?": "El fotoenvejecimiento es el envejecimiento prematuro de la piel causado por la exposición al sol. Puedes prevenir el fotoenvejecimiento protegiéndote del sol, usando protector solar diariamente y evitando el bronceado en interiores.",
                "¿Es necesario protegerse del sol en climas fríos y nublados?": "Sí, es importante protegerse del sol incluso en climas fríos y nublados, ya que los rayos UV pueden penetrar las nubes y causar daño a la piel. Usa protector solar y ropa protectora incluso en días nublados o fríos.",
                "¿El uso de maquillaje con FPS es suficiente protección solar?": "Si bien el maquillaje con FPS puede proporcionar cierta protección contra los rayos UV, no suele ser suficiente por sí solo. Es importante usar protector solar debajo del maquillaje y reaplicarlo según sea necesario para una protección adecuada.",
                "¿El sol puede causar manchas en la piel y decoloración?": "Sí, la exposición al sol puede causar manchas en la piel y decoloración, especialmente en áreas expuestas como la cara, el cuello y las manos. Es importante proteger la piel del sol y tratar las manchas existentes con productos despigmentantes según las recomendaciones de un dermatólogo.",
                "¿Cómo puedo proteger a los niños del sol?": "Puedes proteger a los niños del sol limitando su tiempo de exposición, usando ropa protectora y sombreros, aplicando protector solar y evitando el bronceado en interiores. También es importante enseñarles sobre la importancia de proteger su piel del sol desde una edad temprana.",
                "¿El sol puede causar problemas oculares como cataratas y degeneración macular?": "Sí, la exposición prolongada al sol puede aumentar el riesgo de problemas oculares como cataratas y degeneración macular. Usa gafas de sol con protección UV para proteger tus ojos del daño solar.",
                "¿El sol puede afectar la producción de colágeno en la piel?": "Sí, la exposición al sol puede afectar la producción de colágeno en la piel y contribuir al envejecimiento prematuro. Es importante protegerse del sol y usar productos que estimulen la producción de colágeno para mantener la piel firme y elástica.",
                "¿Qué precauciones debo tomar si tengo un lunar grande o atípico?": "Si tienes un lunar grande o atípico, es importante vigilarlo de cerca y consultar a un dermatólogo si experimentas cambios en su tamaño, forma, color o textura. El médico puede realizar una biopsia para evaluar cualquier cambio sospechoso.",
                "¿Qué es la dermatitis solar y cómo se trata?": "La dermatitis solar es una reacción cutánea inflamatoria causada por la exposición excesiva al sol. Se manifiesta como enrojecimiento, hinchazón y picazón en la piel. Se puede tratar con compresas frías, lociones calmantes y corticosteroides tópicos según lo recomendado por un médico.",
                "¿El sol puede causar daño a largo plazo en la piel?": "Sí, la exposición al sol sin protección puede causar daño a largo plazo en la piel, incluido el envejecimiento prematuro, el cáncer de piel y otros problemas cutáneos como arrugas, manchas oscuras y pérdida de elasticidad.",
                "¿Es seguro exponerse al sol durante el embarazo?": "Si bien es importante obtener vitamina D durante el embarazo, es importante proteger la piel del sol para prevenir el melasma y otras afecciones de la piel asociadas con el embarazo. Consulta a tu médico sobre cómo obtener la cantidad adecuada de vitamina D de manera segura.",
                "¿Qué es la hipersensibilidad al sol y cómo se trata?": "La hipersensibilidad al sol, también conocida como erupción solar o alergia al sol, es una reacción cutánea anormal causada por la exposición al sol. Se manifiesta como enrojecimiento, hinchazón y picazón en la piel expuesta al sol. Se puede tratar con antihistamínicos, corticosteroides tópicos y evitar la exposición al sol.",
                "¿El sol puede empeorar las cicatrices y marcas en la piel?": "Sí, la exposición al sol puede empeorar las cicatrices y marcas en la piel, haciendo que se vuelvan más visibles y oscuras. Es importante proteger las cicatrices del sol y usar protector solar para evitar el empeoramiento.",
                "¿Cómo puedo proteger mis labios del sol?": "Puedes proteger tus labios del sol usando bálsamo labial con FPS y gafas de sol con protección UV. Busca bálsamos labiales que contengan ingredientes hidratantes y protectores solares para mantener tus labios suaves y protegidos del daño solar.",
                "¿El sol puede afectar el sistema inmunológico de la piel?": "Sí, la exposición prolongada al sol puede afectar el sistema inmunológico de la piel y aumentar el riesgo de cáncer de piel y otros problemas cutáneos. Es importante protegerse del sol y evitar la exposición prolongada sin protección.",
                "¿Qué es la hiperpigmentación y cómo se trata?": "La hiperpigmentación es una afección cutánea que se manifiesta como manchas oscuras o áreas de pigmentación irregular en la piel. Se puede tratar con productos despigmentantes, peelings químicos, láseres y otras terapias según las recomendaciones de un dermatólogo.",
                "¿El sol puede causar alergias cutáneas como la urticaria solar?": "Sí, la exposición al sol puede desencadenar alergias cutáneas como la urticaria solar, que se manifiesta como ronchas rojas y picazón en la piel expuesta al sol. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede empeorar las condiciones de la piel como el acné rosácea?": "Sí, la exposición al sol puede empeorar las condiciones de la piel como el acné rosácea, causando enrojecimiento y brotes. Es importante proteger la piel del sol y evitar la exposición prolongada sin protección.",
                "¿Qué es el golpe de calor y cómo puedo prevenirlo durante la exposición al sol?": "El golpe de calor es una afección grave causada por la exposición excesiva al sol y el calor. Para prevenirlo, busca sombra, mantente hidratado, usa ropa ligera y evita la exposición al sol durante las horas más calurosas del día.",
                "¿El sol puede causar queratitis y otras lesiones oculares?": "Sí, la exposición prolongada al sol puede causar queratitis y otras lesiones oculares, incluida la catarata y la degeneración macular. Usa gafas de sol con protección UV para proteger tus ojos del daño solar y consulta a un oftalmólogo si experimentas síntomas.",
                "¿El sol puede afectar la cicatrización de las heridas y quemaduras en la piel?": "Sí, la exposición al sol puede afectar la cicatrización de las heridas y quemaduras en la piel, haciendo que las cicatrices sean más visibles y oscuras. Es importante proteger las heridas del sol y seguir las recomendaciones de cuidado de heridas para una cicatrización adecuada.",
                "¿El sol puede causar sensibilidad y picazón en la piel?": "Sí, la exposición al sol puede causar sensibilidad y picazón en la piel, especialmente en personas con piel sensible o propensa a alergias solares. Evita la exposición al sol y usa protector solar y ropa protectora para proteger tu piel de irritaciones.",
                "¿El uso de filtros solares en la ropa es efectivo para protegerse del sol?": "Sí, el uso de ropa con filtros solares integrados puede proporcionar una capa adicional de protección contra los rayos UV. Busca ropa con factor de protección solar (FPS) incorporado para una protección adicional, especialmente en áreas expuestas como la espalda, los hombros y las piernas.",
                "¿La exposición al sol puede causar cáncer de labio?": "Sí, la exposición al sol puede aumentar el riesgo de cáncer de labio, especialmente en personas con labios sensibles o que pasan mucho tiempo al aire libre sin protección. Usa bálsamo labial con FPS y gafas de sol con protección UV para proteger tus labios del daño solar.",
                "¿Qué es la fototerapia y cómo se utiliza para tratar afecciones de la piel?": "La fototerapia es un tratamiento médico que utiliza luz ultravioleta (UV) para tratar afecciones de la piel como psoriasis, eczema y vitiligo. Se administra bajo la supervisión de un dermatólogo y puede ayudar a reducir la inflamación y mejorar la apariencia de la piel.",
                "¿La exposición al sol puede afectar la producción de vitamina D en el cuerpo?": "Sí, la exposición al sol es una de las principales fuentes de vitamina D en el cuerpo. La luz solar activa la producción de vitamina D en la piel, que es importante para la salud ósea y el sistema inmunológico.",
                "¿El sol puede afectar la función de barrera de la piel?": "Sí, la exposición prolongada al sol puede dañar la barrera natural de la piel y afectar su capacidad para retener la humedad y protegerse de las bacterias y los irritantes externos. Es importante proteger la piel del sol y usar productos hidratantes para mantenerla saludable y protegida.",
                "¿Qué es la aceleración del envejecimiento inducido por el sol?": "La aceleración del envejecimiento inducido por el sol es el proceso de envejecimiento prematuro de la piel causado por la exposición crónica al sol. Se manifiesta como arrugas, manchas oscuras, pérdida de elasticidad y otros signos de envejecimiento prematuro.",
                "¿El sol puede causar daño a la piel en días nublados?": "Sí, los rayos UV pueden penetrar las nubes y causar daño a la piel incluso en días nublados. Es importante protegerse del sol con protector solar y ropa protectora, incluso cuando el cielo está cubierto.",
                "¿Qué es la fotoalergia y cómo se trata?": "La fotoalergia es una reacción alérgica anormal causada por la exposición al sol. Se manifiesta como enrojecimiento, picazón y ampollas en la piel expuesta al sol. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede causar deshidratación en la piel?": "Sí, la exposición prolongada al sol puede causar deshidratación en la piel al eliminar la humedad natural y aumentar la pérdida de agua a través de la transpiración. Mantente hidratado bebiendo suficiente agua y usando productos hidratantes para mantener la piel suave y flexible.",
                "¿Qué es la fotosensibilidad y cómo se trata?": "La fotosensibilidad es una reacción anormal de la piel a la luz solar. Se manifiesta como enrojecimiento, inflamación, ampollas y descamación en la piel expuesta al sol. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede afectar la pigmentación de la piel?": "Sí, la exposición al sol puede afectar la pigmentación de la piel, causando manchas oscuras, manchas solares y decoloración en áreas expuestas. Usa protector solar y ropa protectora para prevenir cambios en la pigmentación de la piel.",
                "¿Qué es la radiación UV y cómo afecta a la piel?": "La radiación ultravioleta (UV) es una forma de energía emitida por el sol que puede dañar la piel y causar cáncer de piel. La radiación UV se clasifica en UVA, UVB y UVC, y puede penetrar en las capas profundas de la piel, causando daño celular y envejecimiento prematuro.",
                "¿El sol puede causar alergias solares como la lucitis solar?": "Sí, la exposición al sol puede desencadenar alergias solares como la lucitis solar, que se manifiesta como picazón, enrojecimiento y erupciones en la piel expuesta al sol. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede aumentar el riesgo de problemas vasculares como las arañas vasculares?": "Sí, la exposición prolongada al sol puede aumentar el riesgo de problemas vasculares como las arañas vasculares y las venas varicosas. Usa protector solar y ropa protectora para prevenir daños en la piel y problemas circulatorios asociados con la exposición al sol.",
                "¿El sol puede causar sensibilidad y dolor en la piel?": "Sí, la exposición al sol puede causar sensibilidad y dolor en la piel, especialmente en personas con piel sensible o propensa a quemaduras solares. Evita la exposición al sol y usa protector solar y ropa protectora para proteger tu piel de irritaciones.",
                "¿El sol puede empeorar las condiciones de la piel como la rosácea?": "Sí, la exposición al sol puede empeorar las condiciones de la piel como la rosácea, causando enrojecimiento y brotes. Es importante proteger la piel del sol y evitar la exposición prolongada sin protección.",
                "¿Qué es la fotosensibilización y cómo se trata?": "La fotosensibilización es una reacción cutánea anormal causada por la exposición al sol o la luz artificial. Se manifiesta como enrojecimiento, inflamación y ampollas en la piel expuesta. Evita la exposición a la luz y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede causar alergias solares como la polimorfosis lumínica benigna?": "Sí, la exposición al sol puede desencadenar alergias solares como la polimorfosis lumínica benigna, que se manifiesta como erupciones y picazón en la piel expuesta. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede aumentar el riesgo de problemas capilares como la cuperosis?": "Sí, la exposición prolongada al sol puede aumentar el riesgo de problemas capilares como la cuperosis, que se manifiesta como vasos sanguíneos visibles en la cara y enrojecimiento. Usa protector solar y ropa protectora para prevenir daños en la piel y problemas capilares asociados con la exposición al sol.",
                "¿El sol puede afectar la salud de las uñas?": "Sí, la exposición al sol puede afectar la salud de las uñas, haciendo que se vuelvan quebradizas, débiles o amarillentas. Usa protector solar y guantes para proteger tus manos del sol y mantener la salud de las uñas.",
                "¿Qué es la queratosis seborreica y cómo se trata?": "La queratosis seborreica es una lesión cutánea benigna que se manifiesta como manchas marrones o negras en la piel. Se puede tratar con crioterapia, extirpación quirúrgica o láser según las recomendaciones de un dermatólogo.",
                "¿El sol puede causar cambios en la textura y elasticidad de la piel?": "Sí, la exposición al sol puede causar cambios en la textura y elasticidad de la piel, haciéndola más áspera, seca y menos flexible. Usa protector solar y productos hidratantes para mantener la piel suave y elástica.",
                "¿El sol puede afectar la función de barrera de la piel?": "Sí, la exposición prolongada al sol puede dañar la barrera natural de la piel y afectar su capacidad para retener la humedad y protegerse de las bacterias y los irritantes externos. Es importante proteger la piel del sol y usar productos hidratantes para mantenerla saludable y protegida.",
                "¿Qué es la aceleración del envejecimiento inducido por el sol?": "La aceleración del envejecimiento inducido por el sol es el proceso de envejecimiento prematuro de la piel causado por la exposición crónica al sol. Se manifiesta como arrugas, manchas oscuras, pérdida de elasticidad y otros signos de envejecimiento prematuro.",
                "¿El sol puede causar daño a la piel en días nublados?": "Sí, los rayos UV pueden penetrar las nubes y causar daño a la piel incluso en días nublados. Es importante protegerse del sol con protector solar y ropa protectora, incluso cuando el cielo está cubierto.",
                "¿Qué es la fotoalergia y cómo se trata?": "La fotoalergia es una reacción alérgica anormal causada por la exposición al sol. Se manifiesta como enrojecimiento, picazón y ampollas en la piel expuesta al sol. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede afectar la pigmentación de la piel?": "Sí, la exposición al sol puede afectar la pigmentación de la piel, causando manchas oscuras, manchas solares y decoloración en áreas expuestas. Usa protector solar y ropa protectora para prevenir cambios en la pigmentación de la piel.",
                "¿Qué es la radiación UV y cómo afecta a la piel?": "La radiación ultravioleta (UV) es una forma de energía emitida por el sol que puede dañar la piel y causar cáncer de piel. La radiación UV se clasifica en UVA, UVB y UVC, y puede penetrar en las capas profundas de la piel, causando daño celular y envejecimiento prematuro.",
                "¿El sol puede causar alergias solares como la lucitis solar?": "Sí, la exposición al sol puede desencadenar alergias solares como la lucitis solar, que se manifiesta como picazón, enrojecimiento y erupciones en la piel expuesta al sol. Evita la exposición al sol y consulta a un dermatólogo para recibir tratamiento.",
                "¿El sol puede aumentar el riesgo de problemas vasculares como las arañas vasculares?": "Sí, la exposición prolongada al sol puede aumentar el riesgo de problemas vasculares como las arañas vasculares y las venas varicosas. Usa protector solar y ropa protectora para prevenir daños en la piel y problemas circulatorios asociados con la exposición al sol."
            
            }

            # Buscar la pregunta en las preguntas frecuentes y devolver la respuesta asociada con mayor similitud
            for key, value in preguntas_frecuentes.items():
                similitud = calcular_similitud(pregunta, key)
                if similitud > max_similitud:
                    max_similitud = similitud
                    respuesta = value
            
            # Si la similitud máxima es baja, mostrar un mensaje predeterminado
            if max_similitud < 0.5:
                respuesta = "Lo siento, no puedo responder esa pregunta en este momento. Por favor, intenta con otra pregunta relacionada con la prevención solar y el cuidado de la piel."

            return respuesta

        # Interfaz de usuario de Streamlit
        st.title("DermIA: ")
        pregunta_usuario = st.text_input("¡Hola! Soy DermIA, ¿en que puedo ayudarte?")

        if pregunta_usuario:
            respuesta_chatbot = responder_pregunta(pregunta_usuario)
            st.text(respuesta_chatbot)