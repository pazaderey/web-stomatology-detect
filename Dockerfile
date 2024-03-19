FROM tensorflow/tensorflow:2.15.0

WORKDIR /home/web-stomatology

COPY requirements.txt /home/web-stomatology

RUN pip install -r requirements.txt --ignore-installed

COPY . /home/web-stomatology

EXPOSE 5000

CMD ["python", "main.py"]
