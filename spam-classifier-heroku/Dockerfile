FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

EXPOSE 8080



RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


COPY ./app /app/app


ENTRYPOINT ["python3","app/app.py"]
