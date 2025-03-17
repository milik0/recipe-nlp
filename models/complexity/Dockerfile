FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PYTHONPATH="/home/generator/.local/lib/python3/site-packages/"

USER generator
WORKDIR /home/generator

COPY requirements.txt ./

RUN python3 -m venv .venv \
    && . .venv/bin/activate \
    && pip install --upgrade pip \
    && pip install pipenv \
    && pipenv install --deploy --ignore-pipfile

COPY . .
ENTRYPOINT [ "pipenv", "run", "python3" ]
CMD ["pipenv", "run", "python3", "src/app.py"]
