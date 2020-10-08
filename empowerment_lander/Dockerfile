FROM tensorflow/tensorflow:1.14.0-py3

RUN apt-get -y update && apt-get -y install ffmpeg && apt-get install -y git

RUN git clone https://github.com/openai/baselines.git
RUN pip install -e baselines
RUN pip install gym[box2d]