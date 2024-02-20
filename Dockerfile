FROM python:3.8
RUN pip install pandas scikit-learn==1.2.2 streamlit numpy  tensorflow scipy protobuf keras Pillow
COPY src/* /app/
COPY model/benigno_vs_maligno_modelo.pkl /app/model/benigno_vs_maligno_modelo.pkl
WORKDIR /app
ENTRYPOINT [ "streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]