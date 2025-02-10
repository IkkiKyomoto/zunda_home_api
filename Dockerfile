FROM python:3.9
WORKDIR /app
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt
COPY ./app /code/app
ENV PYTHONPATH="/code/app:${PYTHONPATH}"
RUN cd /code/app
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]
EXPOSE 80
