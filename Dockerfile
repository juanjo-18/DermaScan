FROM python:3.8
RUN pip install pandas scikit-learn==1.2.2 streamlit==1.26 numpy tensorflow scipy protobuf keras Pillow st-files-connection plotly==5.16.0 s3fs boto3 hydralit requests beautifulsoup4 matplotlib seaborn
COPY src/* /app/
COPY model/* /app/model/*
WORKDIR /app
ENTRYPOINT [ "streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]