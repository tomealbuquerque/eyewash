FROM python:3.9

RUN apt-get update; apt-get install

RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/appl/"

WORKDIR /appl/
