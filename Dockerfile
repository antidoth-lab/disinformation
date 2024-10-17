FROM python:3.11

WORKDIR /src

COPY ./requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# api folder
COPY ./api/main.py /src/api/main.py
COPY ./api/service_runner.py /src/api/service_runner.py

# data folder
COPY ./data/bad-words-and-phrases.csv /src/data/bad-words-and-phrases.csv

# service folder
COPY ./services/__init__.py /src/services/__init__.py
COPY ./services/emotional_clickbait.py /src/services/emotional_clickbait.py
COPY ./services/polarization.py /src/services/polarization.py
COPY ./services/trolling.py /src/services/trolling.py
COPY ./services/whataboutism.py /src/services/whataboutism.py

# app folder
COPY ./typings/__init__.py /src/typings/__init__.py
COPY ./typings/type_annotations.py /src/typings/type_annotations.py

# .env file
COPY ./.env /src/.env

ENV PYTHONPATH=/src
EXPOSE 8080

CMD ["fastapi", "run", "api/main.py", "--proxy-headers", "--port", "8080"]
