#!/usr/bin/env python
# coding: utf-8

import os
import jwt
import json
import time
import logging
import requests
from opensearchpy import OpenSearch
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import TextLoader, S3DirectoryLoader
from langchain.chains import LLMChain
from yagpt import YandexGPTEmbeddings, YandexLLM
from flask import Flask, request, Response
from multiprocessing import Process


def read_json(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


ROOT_PATH = '.'
PORT = 40000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BULK_SIZE = 1000
CA_PATH = f'{ROOT_PATH}/.opensearch/root.crt'
creds = read_json(f'{ROOT_PATH}/configs/credentials.json')
DB_USER = creds['db_user']
DB_PASS = creds['db_password']
DB_HOSTS = creds['db_hosts']
LLM_SERVICE_ACCOUNT_ID = creds['service_account_id']
LLM_KEY_ID = creds['key_id']
LLM_PRIVATE_KEY = creds['private_key']
S3_BUCKET = creds['bucket']
S3_KEY_ID = creds['aws_access_key_id']
S3_SECRET_KEY = creds['aws_secret_access_key']
S3_ENDPOINT_URL= creds['endpoint_url']
FOLDER_ID = creds['folder_id']


class BotChain():
    def __init__(self, llm_service_account_id, llm_private_key, llm_key_id,
                 bucket, s3_key_id, s3_secret_key, s3_endpoint_url,
                 chunk_size, chunk_overlap, 
                 db_hosts, db_user, db_pass, ca_path, bulk_size,
                 folder_id):
        self.token = self.ya_token(
            llm_service_account_id, 
            llm_private_key, 
            llm_key_id,
            lag=3600
        )
        self.docsearch = self.db_docsearch(
            bucket, s3_key_id, s3_secret_key, s3_endpoint_url,
            chunk_size, chunk_overlap, 
            db_hosts, db_user, db_pass, ca_path, bulk_size,
            folder_id
        )
        self.folder_id = folder_id

    def db_connect(self, db_hosts, db_user, db_pass, ca_path):
        conn = OpenSearch(
            db_hosts,
            http_auth=(db_user, db_pass),
            use_ssl=True,
            verify_certs=True,
            ca_certs=ca_path
        )
        print('connection:', conn.info())
        return conn

    def ya_token(self, llm_service_account_id, llm_private_key, llm_key_id, lag):
        now = int(time.time())
        payload = {
            'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            'iss': llm_service_account_id,
            'iat': now,
            'exp': now + lag
        }
        encoded_token = jwt.encode(
            payload,
            llm_private_key,
            algorithm='PS256',
            headers={'kid': llm_key_id}
        )
        url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
        r = requests.post(
            url,  
            headers={'Content-Type': 'application/json'},
            json={'jwt': encoded_token}
        ).json()
        token = r['iamToken']
        return token

    def db_docsearch(self, bucket, s3_key_id, s3_secret_key, s3_endpoint_url,
                     chunk_size, chunk_overlap, 
                     db_hosts, db_user, db_pass, ca_path, bulk_size, 
                     folder_id):
        loader = S3DirectoryLoader(
            S3_BUCKET, 
            aws_access_key_id=s3_key_id, 
            aws_secret_access_key=s3_secret_key,
            endpoint_url=s3_endpoint_url
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        embeddings = YandexGPTEmbeddings(iam_token=self.token, folder_id=folder_id)
        docsearch = OpenSearchVectorSearch.from_documents(
            docs,
            embeddings,
            opensearch_url=db_hosts[0],
            http_auth=(db_user, db_pass),
            use_ssl=True,
            verify_certs=True,
            ca_certs=ca_path,
            engine='lucene',
            bulk_size=bulk_size
        )
        return docsearch

    def db_simularity_search(self, query, k=2):
        query_docs = self.docsearch.similarity_search(query, k=k)
        return query_docs
        
    def ya_chain(self, temperature, instructions):
        llm = YandexLLM(
            iam_token=self.token,
            folder_id=self.folder_id,
            temperature=temperature,
            instruction_text=instructions
        )
        document_prompt = langchain.prompts.PromptTemplate(
            input_variables=['page_content'], 
            template='{page_content}'
        )
        document_variable_name = 'context'
        prompt_override = """
            Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
            Текст:
            -----
            {context}
            -----
            Вопрос:
            {query}
            """
        prompt = langchain.prompts.PromptTemplate(
            template=prompt_override,
            input_variables=['context', 'query']
        )
        llm_chain = langchain.chains.LLMChain(
            llm=llm, 
            prompt=prompt
        )
        chain = langchain.chains.combine_documents.stuff.StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        return chain


def resp(code, data):
    return Response(status=code, mimetype='application/json', response=json.dumps(data))


def theme_validate(fields, chk_str=True):
    errors = []
    jsn = request.get_json()
    if jsn is None:
        errors.append('no JSON sent, check Content-Type header')
        return (None, errors)
    for field_name in fields:
        if field_name not in jsn.keys():
            errors.append('field {} is missing'.format(field_name))
    return (jsn, errors)


LOG_PATH = 'logs'
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{LOG_PATH}/server.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
msg = 'Bot server started, logger initialized'
LOGGER.info(msg)

BOTCHAIN = BotChain(
    llm_service_account_id=LLM_SERVICE_ACCOUNT_ID, 
    llm_private_key=LLM_PRIVATE_KEY,
    llm_key_id=LLM_KEY_ID,
    db_hosts=DB_HOSTS, 
    db_user=DB_USER, 
    db_pass=DB_PASS, 
    ca_path=CA_PATH,
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    bulk_size=BULK_SIZE,
    bucket=S3_BUCKET, 
    s3_key_id=S3_KEY_ID, 
    s3_secret_key=S3_SECRET_KEY, 
    s3_endpoint_url=S3_ENDPOINT_URL,
    folder_id=FOLDER_ID
)
CHAIN, DOCS = None, None
msg = 'BotChain started, documents uploaded to database'
LOGGER.info(msg)


app = Flask(__name__)


@app.route('/datainfo', methods=['GET'])
def data_config():
    data = {
        'bucket': creds['bucket'], 
        'bucket_info': creds['bucket_info']
    }
    return resp(200, {'data': data})


@app.route('/creds', methods=['GET'])
def server_config():
    return resp(200, {'data' : list(creds.keys())})


@app.route('/logs', methods=['GET'])
def server_logs():
    with open(f'{LOG_PATH}/server.log') as file:
        logs = file.readlines()
    return resp(200, {'data' : logs})


@app.route('/init', methods=['POST'])
def init_chain():
    fields = ['temperature', 'instructions']
    (jsn, errors) = theme_validate(fields)
    if errors:
        return resp(400, {'errors': errors})
    instructions = jsn['instructions']
    temperature = jsn['temperature']
    global CHAIN
    CHAIN = BOTCHAIN.ya_chain(temperature, instructions)
    msg = 'Chain for instructions -{}- done'.format(
        instructions.replace('\n', '').replace('\t', '')
    )
    LOGGER.info(msg)
    return resp(200, {'result' : msg})


@app.route('/search', methods=['POST'])
def search_db():
    fields = ['query', 'k']
    (jsn, errors) = theme_validate(fields)
    if errors:
        return resp(400, {'errors': errors})
    query, k = jsn['query'], jsn['k']
    global DOCS
    DOCS = BOTCHAIN.db_simularity_search(query=query, k=k)
    msg = 'Search for query -{}- done, k = {}'.format(
        query,
        k
    )
    LOGGER.info(msg)
    return resp(200, {'result' : msg})


@app.route('/ask', methods=['POST'])
def ask_chain():
    fields = ['query']
    (jsn, errors) = theme_validate(fields)
    if errors:
        return resp(400, {'errors': errors})
    query = jsn['query']
    response = CHAIN.run(input_documents=DOCS, query=query)
    msg = 'Q - {} | A - {}'.format(
        query,
        response
    )
    LOGGER.info(msg)
    return resp(200, {'answer' : response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
