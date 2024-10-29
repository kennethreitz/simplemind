FROM python:3.12.0

RUN apt-get update -y && apt-get upgrade -y
RUN pip install --upgrade pip

COPY requirements.txt /src/requirements.txt

WORKDIR /src

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]