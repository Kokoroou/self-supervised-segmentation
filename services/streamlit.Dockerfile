FROM python3.8-slim

RUN apt update

RUN pip3 install --upgrade pip

COPY requirements_pytorch.txt /tmp/requirements_pytorch.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements_pytorch.txt

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY . /app
WORKDIR /app/self-supervised-segmentation

RUN pip3 install -e .

CMD streamlit run /app/self-supervised-segmentation/services/streamlit/app.py --server.port 8585