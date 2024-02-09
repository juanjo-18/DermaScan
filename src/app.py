import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta


# Función para seleccionar fecha con un selector de calendario
def date_selector(label, default_date):
    selected_date = st.date_input(label, default_date)
    return datetime.combine(selected_date, datetime.min.time())

# Función para calcular días de semana y fines de semana
def calculate_weekdays_and_weekends(start_date, end_date):
    weekdays = 0
    weekends = 0

    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # 0-4 representan días de semana (lunes a viernes)
            weekdays += 1
        else:
            weekends += 1
        current_date += timedelta(days=1)

    return weekdays, weekends

# Función para calcular el número de noches de lunes a viernes entre dos fechas
def calcular_noches_lunes_a_viernes(fecha_inicio, fecha_fin):
    # Inicializar el contador de noches
    noches = 0

    # Definir el delta de un día
    un_dia = timedelta(days=1)

    # Iterar sobre cada día entre las fechas de inicio y fin
    while fecha_inicio <= fecha_fin:
        # Verificar si el día actual es de lunes a viernes (días laborables)
        if fecha_inicio.weekday() < 5:  # 0 es lunes, 1 es martes, ..., 4 es viernes
            noches += 1

        # Avanzar al siguiente día
        fecha_inicio += un_dia

    return noches

# Función para calcular el número de noches de sábado o domingo entre dos fechas
def calcular_noches_sabado_domingo(fecha_inicio, fecha_fin):
    # Inicializar el contador de noches
    noches_sabado_domingo = 0

    # Definir el delta de un día
    un_dia = timedelta(days=1)

    # Iterar sobre cada día entre las fechas de inicio y fin
    while fecha_inicio <= fecha_fin:
        # Verificar si el día actual es sábado o domingo
        if fecha_inicio.weekday() in [5, 6]:  # 5 es sábado, 6 es domingo
            noches_sabado_domingo += 1

        # Avanzar al siguiente día
        fecha_inicio += un_dia

    return noches_sabado_domingo


# Página principal
def main():
    # Encabezado
    st.title("Aplicación de Hotel")
    st.subheader("Trabajo realizado por Juanjo")
    st.markdown("### Predicción de coste de habitación")  # Subtítulo

    # Selector de fecha de entrada
    st.subheader("Fecha de Entrada")
    start_date = date_selector("Selecciona la fecha de entrada:", datetime.now())

    # Selector de fecha de salida
    st.subheader("Fecha de Salida")
    end_date = date_selector("Selecciona la fecha de salida:", datetime.now() + timedelta(days=1))

    # Desplegable para el tipo de habitación
    reserved_room_type = st.selectbox("Tipo de Habitación, Siendo la A la peor y la G la mejor.", ['A', 'B', 'C','D','E','F','G'])

    # Desplegable para el numero de adultos
    adults = st.selectbox("Numero de adultos", [1, 2, 3, 4, 5])


    # Desplegable para el numero de adultos
    meal = st.selectbox("Tipo de comida", ["HB - Desayuno y otra comida", "BB - Desayuno", "SC - Sin paquete de comida", "FB-Desayuno, almuerzo y cenaB"])

    # Calcular la duración en días
    duration = (end_date - start_date).days

    # Variable para almacenar la fecha del mes
    month_date = start_date.strftime("%B %Y")  # Formato: Nombre del mes Año

    # Variables para calcular
    agent= "129."
    # Obtener la fecha y hora actual
    now = datetime.now()
    # Formatear la fecha y hora como una cadena en el formato "YYYY-MM-DD"
    formatted_date = now.strftime("%Y-%m-%d")
    reservation_status_date= formatted_date
    arrival_date_week_number = start_date.isocalendar()[1]
    fecha_formateada = datetime.strptime(formatted_date, "%Y-%m-%d") 
    diferencia= start_date - fecha_formateada
    lead_time=diferencia.days
    arrival_date_day_of_month=start_date.day
    stays_in_week_nights=calcular_noches_lunes_a_viernes(start_date, end_date)
    stays_in_weekend_nights=calcular_noches_sabado_domingo(start_date, end_date)
    arrival_date_month=start_date.month

    # Botón para iniciar el cálculo
    if st.button("Empezar"):
        clf = joblib.load("model/hotel_model.pkl")
        X = pd.DataFrame([[lead_time,arrival_date_month,arrival_date_week_number,arrival_date_day_of_month,stays_in_weekend_nights,stays_in_week_nights,adults,meal,reserved_room_type,agent,reservation_status_date]],columns=["lead_time","arrival_date_month","arrival_date_week_number","arrival_date_day_of_month","stays_in_weekend_nights","stays_in_week_nights","adults","meal","reserved_room_type","agent","reservation_status_date"])
        
        X.replace(["A", "B", "C", "D", "E", "F", "G"], [0, 1, 2, 3, 4, 5, 6], inplace=True)
        X.replace(["HB - Desayuno y otra comida", "BB - Desayuno", "SC - Sin paquete de comida", "FB - Desayuno, almuerzo y cena"], [0, 1, 2, 3], inplace=True)
        X["agent"].replace(X["agent"].unique(), np.arange(len(X["agent"].unique())), inplace=True)
        X["reservation_status_date"].replace(X["reservation_status_date"].unique(), np.arange(len(X["reservation_status_date"].unique())), inplace=True)

        prediction = clf.predict(X)[0]

        # Mostrar la duración en días
        st.write(f"Duración de la estadía: {duration} días")

        precio_total=duration*prediction
        # Mostrar el número de días de semana y fines de semana
        st.success(f"Precio por noche: {prediction}, precio total: {precio_total}")

        

if __name__ == "__main__":
    main()
