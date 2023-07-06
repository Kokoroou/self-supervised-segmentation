FROM python:3.8-slim

RUN apt update && apt install ffmpeg libsm6 libxext6 -y

RUN pip3 install --no-cache-dir --upgrade pip

COPY ./requirements-pytorch.txt /Workspace/requirements-pytorch.txt
RUN pip3 install --no-cache-dir -r /Workspace/requirements-pytorch.txt

COPY ./requirements.txt /Workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /Workspace/requirements.txt

ADD . /Workspace

WORKDIR /Workspace

RUN pip3 install --no-cache-dir -e .

WORKDIR /Workspace/self-supervised-segmentation/services/streamlit

CMD streamlit run ./Home.py --server.port 8585