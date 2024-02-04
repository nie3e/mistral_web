FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
RUN apt-get -y update
RUN apt-get -y install git
RUN mkdir /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN apt-get -y purge --auto-remove git
COPY src /app/src
RUN pip install -e /app/src