FROM registry.blitzhub.io/fc_flask

COPY . /app/fc_app

RUN pip3 install -r ./app/fc_app/requirements.txt
