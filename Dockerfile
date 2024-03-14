FROM python:3.9

WORKDIR /home/web-stomatology

COPY requirements.txt /home/web-stomatology

RUN pip3 install --upgrade pip -r requirements.txt

COPY . /home/web-stomatology

EXPOSE 5000
