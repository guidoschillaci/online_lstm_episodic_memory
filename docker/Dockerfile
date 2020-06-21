FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN pip install pandas scikit-learn doepy seaborn matplotlib scikit-image

RUN useradd -ms /bin/bash green
USER green
WORKDIR /home/green
