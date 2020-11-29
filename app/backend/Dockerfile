FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /backend

COPY requirements.txt /backend

WORKDIR /backend


RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . /backend

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]