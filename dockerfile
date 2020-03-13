FROM python:3.7

COPY . /home/app

RUN apt-get -y update
RUN pip install --upgrade pip


RUN pip install scikit-learn==0.22.2
RUN pip install tensorflow==1.15.2
RUN pip install tensorflow-hub==0.7.0
RUN pip install sentencepiece==0.1.85
RUN pip install Flask==1.1.1

EXPOSE 5000

WORKDIR /home/app

CMD python main.py