FROM python:3.8-slim

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
RUN pip install torch==1.8.1 gunicorn==20.1.0 tokenizers==0.10.2 torch-geometric==1.7.0 Flask==2.0.0

RUN mkdir /app
RUN mkdir /app/resources
RUN mkdir /app/templates
WORKDIR /app

ADD resources /app/resources
ADD templates /app/templates
COPY *.py /app/

ENTRYPOINT gunicorn app:app -b 0.0.0.0:5000