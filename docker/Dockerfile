from python:3.9.7

WORKDIR /server

ADD train.py train.py
ADD predict-server.py predict-server.py
ADD requirements.txt requirements.txt
ADD /weights/level1.hdf5 ./weights/level1.hdf5
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python", "./predict-server.py" , "level1", "--cpu"]
EXPOSE 36296/TCP
