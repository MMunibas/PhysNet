
FROM tensorflow/tensorflow:1.13.2
RUN apt-get update
EXPOSE 8501
WORKDIR /
COPY . .

RUN apt install wget -y
RUN wget -O sn2_reactions.npz https://zenodo.org/record/2605341/files/sn2_reactions.npz?download=1

RUN apt-get install zlib1g-dev libjpeg8-dev -y

RUN apt-get install python3-pip -y

RUN pip3 install --upgrade setuptools wheel

RUN pip3 install -r requirements.txt

CMD python3 train.py