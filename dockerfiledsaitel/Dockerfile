FROM cr.yandex/crptfs8246iv37ojkorv/mibadsai:20230920v1

USER root

RUN apt update && apt install curl -y

USER $NB_UID

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt