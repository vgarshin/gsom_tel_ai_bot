#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import jwt
import requests
from opensearchpy import OpenSearch
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from YaGPT import YandexGPTEmbeddings, YandexLLM
from flask import Flask, request, Response
from multiprocessing import Process

PORT = 40000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
BULK_SIZE = 1000
CA_PATH = '.opensearch/root.crt'

def read_credentials(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

creds = read_credentials('credentials.json')
USER = creds['db_user']
PASS = creds['db_password']
HOSTS = creds['db_hosts']
service_account_id = creds['service_account_id']
key_id = creds['key_id']
private_key = creds['private_key']
source_dir = './Teaching essentials'

def start():
    conn = OpenSearch(
      HOSTS,
      http_auth=(USER, PASS),
      use_ssl=True,
      verify_certs=True,
      ca_certs=CA_PATH
    )
    print('connection:', conn.info())

    now = int(time.time())
    payload = {
            'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
            'iss': service_account_id,
            'iat': now,
            'exp': now + 360}
    encoded_token = jwt.encode(
        payload,
        private_key,
        algorithm='PS256',
        headers={'kid': key_id})
    url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
    r = requests.post(
        url,  
        headers={'Content-Type': 'application/json'},
        json={'jwt': encoded_token}
    ).json()
    token = r['iamToken']
    print('token:', token)

    loader = langchain.document_loaders.DirectoryLoader(
        source_dir, 
        glob='*.*',
        silent_errors=True,
        show_progress=True, 
        recursive=True
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    embeddings = YandexGPTEmbeddings(iam_token=token)

    docsearch = OpenSearchVectorSearch.from_documents(
        docs,
        embeddings,
        opensearch_url=HOSTS[0],
        http_auth=(USER, PASS),
        use_ssl=True,
        verify_certs=True,
        ca_certs=CA_PATH,
        engine='lucene',
        bulk_size=BULK_SIZE
    )
    query = 'проектирование домашнего задания'
    print('test query:', query)
    docs = docsearch.similarity_search(query, k=2)
    print('test result:', docs)

    instructions = """
    Вы являетесь помощником преподавателя. 
    Ваша задача - помогать студентам в процессе обучения,
    отвечать на их вопросы, искать нужные ответы в материалах курса.
    """
    llm = YandexLLM(
        iam_token=token,
        instruction_text=instructions
    )
    document_prompt = langchain.prompts.PromptTemplate(
        input_variables=["page_content"], 
        template="{page_content}"
    )
    document_variable_name = "context"
    stuff_prompt_override = """
        Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста.
        Текст:
        -----
        {context}
        -----
        Вопрос:
        {query}
    """
    prompt = langchain.prompts.PromptTemplate(
        template=stuff_prompt_override,
        input_variables=["context", "query"]
    )

    llm_chain = langchain.chains.LLMChain(
        llm=llm, 
        prompt=prompt
    )
    chain = langchain.chains.combine_documents.stuff.StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )

    return docs, chain
    
DOCS, CHAIN = start()

class ReverseProxied(object):
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        scheme = environ.get('HTTP_X_FORWARDED_PROTO')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        return self.app(environ, start_response)

app = Flask(__name__)
app.wsgi_app = ReverseProxied(app.wsgi_app)

def resp(code, data):
    return Response(status=code, mimetype='application/json', response=json.dumps(data))

def theme_validate_ask():
    errors = []
    jsn = request.get_json()
    if jsn is None:
        errors.append('no JSON sent, check Content-Type header')
        return (None, errors)
    for field_name in ['query']:
        if type(jsn.get(field_name)) is not str:
            errors.append('field {} is missing or is not a string'.format(field_name))
    return (jsn, errors)

@app.route('/ask', methods=['POST'])
def ask_chain():
    (jsn, errors) = theme_validate_ask()
    if errors:
        return resp(400, {'errors': errors})
    query = jsn['query']
    response = CHAIN.run(input_documents=DOCS, query=query)
    return resp(200, {'answer' : response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
