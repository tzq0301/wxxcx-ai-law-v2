FROM python:3.9

WORKDIR /code

RUN apt-get update && apt-get -y install cmake protobuf-compiler

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
