FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PYTHONPATH="/home/recommender/.local/lib/python3/site-packages/"

# Install required packages and dependencies
RUN apt-get update --no-install-recommends \
    && apt-get install --no-install-recommends -y \
       python3 git make cron bzip2 tzdata locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && cp /usr/share/zoneinfo/Europe/Paris /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Switch to the recommender user
USER recommender
WORKDIR /home/recommender

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install pipenv and create virtual environment with it
RUN python3 -m venv .venv \
    && . .venv/bin/activate \
    && pip install --upgrade pip \
    && pip install pipenv \
    && pipenv install --deploy --ignore-pipfile

# Copy the rest of the application code
COPY . .

# Default command to run the application
CMD ["pipenv", "run", "python3", "src/__main__.py"]
