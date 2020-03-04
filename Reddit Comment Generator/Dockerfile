FROM python:3.6-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

WORKDIR /
COPY . /

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
RUN pip install -r requirements.txt

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python", "app.py"]
