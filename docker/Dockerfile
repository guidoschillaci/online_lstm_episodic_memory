#FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

FROM tensorflow/tensorflow:2.1.1-gpu

RUN pip install pandas scikit-learn doepy seaborn matplotlib scikit-image

RUN git clone https://github.com/guidoschillaci/online_lstm_episodic_memory.git
# needs Data folder, still not released as open access. Ask authors for permissions

RUN useradd -ms /bin/bash green
USER green
WORKDIR /home/green/online_lstm_episodic_memory
