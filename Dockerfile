FROM python:3.8
RUN pip install pandas scikit-learn==1.3.1 streamlit==1.26 numpy tensorflow scipy protobuf keras Pillow st-files-connection plotly==5.16.0 s3fs
COPY src/* /app/
COPY model/benigno_vs_maligno_modelo.pkl /app/model/benigno_vs_maligno_modelo.pkl
WORKDIR /app
ENTRYPOINT [ "streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]