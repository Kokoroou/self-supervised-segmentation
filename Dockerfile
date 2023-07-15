FROM python:3.9-slim

RUN apt update && apt install ffmpeg libsm6 libxext6 -y

RUN pip3 install --no-cache-dir --upgrade pip

COPY ./services/requirements_pytorch.txt /Workspace/requirements_pytorch.txt
RUN pip3 install --no-cache-dir -r /Workspace/requirements_pytorch.txt

COPY ./services/requirements.txt /Workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /Workspace/requirements.txt

ADD ./services /Workspace

WORKDIR /Workspace

RUN pip3 install --no-cache-dir -e .

WORKDIR /Workspace/ui

CMD streamlit run --server.port 8585 "Home.py"