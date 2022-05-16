FROM python:3.9

RUN apt-get update; apt-get install

COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Streamlit configs
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

ENV PYTHONPATH "${PYTHONPATH}:/appl/"

WORKDIR /appl/