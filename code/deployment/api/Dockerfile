FROM python:3.11.9-slim

WORKDIR /api

COPY ./main.py /api
COPY ./model /api
RUN pip install -r requirements.txt
COPY ./requirements.txt /api
RUN pip install -r requirements.txt
RUN pip freeze > requirements.txt


#COPY ../../models/model.pkl /api


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "3500"]