FROM python:3.8
RUN pip install pandas scikit-learn==1.2.2 streamlit numpy
COPY src/* /app/
COPY model/hotel_model.pkl /app/model/hotel_model.pkl
WORKDIR /app
ENTRYPOINT [ "streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]