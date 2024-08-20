import glob
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr
import json
import re
import copy
import time
import argparse
import shutil
import signal
from openai import AzureOpenAI, OpenAI
import pandas as pd
import numpy as np
import ast
import subprocess
from prompts.preprocess import *
from langchain.document_loaders import TextLoader 
from langchain.indexes import VectorstoreIndexCreator

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
def init_rag(pdf_directory, embedding_model='ada002'):
    os.environ['PYTHONNUMBUFFER'] = '1'
    os.environ['OPENAI_API_TYPE'] ='azure'
    os.environ['OPENAI_API_VERSION'] ='2023-10-01-preview'
    os.environ['OPENAI_API_KEY'] ="1b9af3abe2b34c56abe2e2f0c4f8a60b"
    os.environ['AZURE_OPENAI_ENDPOINI'] ="https://tiancheng.openai.azure.com/"
    pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    all_documents = []
    for pdf_path in pdf_files:
        loader = TextLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_documents)
    embeddings = AzureOpenAIEmbeddings(model="ada002",openai_api_type='azure',azure_endpoint='https://tiancheng.openai.azure.com/',openai_api_key='1b9af3abe2b34c56abe2e2f0c4f8a60b',chunk_size=1)
    db = Chroma.from_documents(docs, embeddings)
    return db

def query_rag(db, query):
    llm=AzureChatOpenAI(azure_deployment='gpt4o',azure_endpoint='https://tiancheng.openai.azure.com/',temperature=0)
    result = db.similarity_search(query, llm=llm)
    return result
def query_online(query):
    pass
# init rag
pdf_directory = './expertknowledge' 
db = init_rag(pdf_directory)
query = "--"
result = query_rag(db, query)
