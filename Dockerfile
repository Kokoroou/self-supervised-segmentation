FROM python:3.9-slim

RUN apt update && apt install ffmpeg libsm6 libxext6 -y

RUN pip3 install --no-cache-dir --upgrade pip

COPY ./services/requirements_pytorch.txt /app/requirements_pytorch.txt
RUN pip3 install --no-cache-dir -r /app/requirements_pytorch.txt

COPY ./services/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

ADD ./services /app

WORKDIR /app

RUN pip3 install --no-cache-dir -e .

WORKDIR /app/ui

CMD streamlit run --server.port 8585 "Home.py"