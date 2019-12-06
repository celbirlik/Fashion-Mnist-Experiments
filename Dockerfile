FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

LABEL maintainer='celbirlik@gmail.com'

WORKDIR /fashion_mnist

ADD . /fashion_mnist

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8888


CMD jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.allow_origin='*'


