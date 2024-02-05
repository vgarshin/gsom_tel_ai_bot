# GSOM TEL YaGPT bot for Teaching essentials course
Experimental AI bot based on [Yandex repository for Retrieval-Augmented Generation](https://github.com/yandex-cloud-examples/yc-yandexgpt-qa-bot-for-docs) with help of Yandex.Cloud services.

## Goal

Run a simple AI bot for Teaching Essentials course based on YandexGPT LLM and OpenSearch database to store context in a vector form (embeddings).

## Tools

[Streamlit](https://streamlit.io/) for the interface of the application, [Flask](https://flask-docs.readthedocs.io/en/latest/) for the backend, [Docker](https://www.docker.com/) to run a services.

## Prerequisites

Data on Teaching Essentials course unpacked to `Teaching essentials` at the server side.

[Yandex Managed Service for OpenSearch](https://cloud.yandex.com/en/docs/managed-opensearch/) instance as a database. [Virtual machine](https://cloud.yandex.com/en/docs/compute/quickstart/) with Docker installed.

Credentials to access database and YandexGPT API in a form of JSON file:

```json
{
   "db_user": "<USER_NAME>", 
   "db_password": "<PASSWORD>", 
   "db_hosts": [
      "<HOST_ADDRESS>"
   ], 
   "service_account_id": "<ID>",
   "key_id": "KEY_ID",
   "private_key": "<PRIVATE_KEY>"
}
```

## Manual

Build an image to run all services, image is based on image for Data Science environment at the platform [GSOM JupyterHub](https://github.com/vgarshin/gsom_jhub_deploy). Command to build:

```bash
sudo docker build -t mibadsaitel dockerfiledsaitel
```

Run service for AI course assistant:

```bash
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 40000:40000 -it mibadsaitel python aihelper.py
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 30000:30000 -it mibadsaitel streamlit run aibothelper.py --server.port 30000 --browser.gatherUsageStats False

```

Run service for AI course home assignment validator:

```bash
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 40001:40001 -it mibadsaitel python aichecker.py
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 30001:30001 -it mibadsaitel streamlit run aibotchecker.py --server.port 30001 --browser.gatherUsageStats False
```

...or use `docker compose` to run all applications at once:

```bash
docker compose up
```
