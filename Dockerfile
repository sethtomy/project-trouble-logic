FROM arm32v7/python:3

COPY app /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["server.py"]