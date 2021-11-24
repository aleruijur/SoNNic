from python:3.7

WORKDIR /server

ADD train.py train.py
ADD predict-server.py predict-server.py
ADD requirements.txt requirements.txt
ADD /weights/demo.hdf5 ./weights/demo.hdf5
RUN pip install -r requirements.txt
CMD [ "python", "./predict-server.py" , "demo", "--cpu"]
EXPOSE 36296/TCP
