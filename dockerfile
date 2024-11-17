FROM python:3-buster
RUN pip install --upgrade pip
WORKDIR /model
COPY ./model/requirements.txt .
RUN pip install -r requirements.txt
COPY ./model .
CMD ["python", "main.py"]
