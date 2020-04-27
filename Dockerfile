FROM tensorflow/tensorflow:latest-py3
COPY app /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["server.py"]