FROM nvidia/cuda:8.0-cudnn5-devel
USER root

RUN apt-get update
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev \
xvfb libav-tools xorg-dev python-opengl libav-tools swig \
python3 python3-pip liblapack-dev \
libopenblas-dev python3-dev gfortran git vim
RUN pip3 install --upgrade pip
RUN pip3 install cython numpy scipy matplotlib
RUN pip3 install --upgrade https://github.com/Theano/Theano/archive/master.zip
RUN pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
RUN pip3 install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip
RUN mkdir ~/gym && cd ~/gym && git clone https://github.com/openai/gym.git && cd gym && pip3 install -e .[atari]
RUN pip3 install jupyter nose
COPY .theanorc /root/
COPY run_jupyter.sh /
COPY jupyter_config.py /root/.jupyter/
RUN mkdir /root/notebooks
WORKDIR /root/notebooks
CMD /run_jupyter.sh
