# GSOM TEL YaGPT bot for Teaching essentials course
Experimental AI bot based on [Yandex repository for Retrieval-Augmented Generation](https://github.com/yandex-cloud-examples/yc-yandexgpt-qa-bot-for-docs) with help of Yandex.Cloud services.

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

```bash
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 40000:40000 -it mibadsaitel python aihelper.py
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 40001:40001 -it mibadsaitel python aichecker.py
```

```bash
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 30000:30000 -it mibadsaitel streamlit run aibothelper.py --server.port 30000 --browser.gatherUsageStats False
sudo docker run -d -v /home/teluser/aibot:/home/jovyan/ -p 30001:30001 -it mibadsaitel streamlit run aibotchecker.py --server.port 30001 --browser.gatherUsageStats False
```
