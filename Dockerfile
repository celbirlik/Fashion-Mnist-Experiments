FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ADD . /fashion_mnist

MAINTAINER Can Elbirlik, celbirlik@gmail.com


RUN pip install --trusted-host pypi.python.org requirements.txt

EXPOSE 8888


CMD jupyter notebook --no-browser --NotebookApp.token=''


