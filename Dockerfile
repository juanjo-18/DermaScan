FROM python:3.8
RUN pip install pandas scikit-learn==1.2.2 streamlit numpy tensorflow==2.11.0 scipy protobuf>=3.9.2
COPY src/* /app/
COPY model/benigno_vs_maligno_modelo.pkl /app/model/benigno_vs_maligno_modelo.pkl
WORKDIR /app
ENTRYPOINT [ "streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]