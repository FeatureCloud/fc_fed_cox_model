FROM python:3.8

RUN apt-get update
RUN apt-get install -y redis-server supervisor nginx

RUN pip3 install --upgrade pip
RUN pip3 install gunicorn

COPY supervisord.conf /supervisord.conf
COPY nginx/default /etc/nginx/sites-available/default
COPY docker-entrypoint.sh /entrypoint.sh

COPY requirements.txt ./app/requirements.txt

RUN pip3 install -r ./app/requirements.txt

RUN pip3 uninstall numpy -y
RUN pip3 install -U numpy

COPY . /app

EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]