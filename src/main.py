from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
import time
import json
import os
import gradio as gr
import uuid
from sparkai.core.messages import ChatMessage, AIMessageChunk
from dwspark.config import Config
from dwspark.models import ChatModel, ImageUnderstanding, Text2Audio, Audio2Text, EmbeddingModel
from PIL import Image
import io
import base64
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity 
import pickle
import re
import time
import numpy as np

SPARKAI_APP_ID = os.environ.get("SPARKAI_APP_ID")
SPARKAI_API_SECRET = os.environ.get("SPARKAI_API_SECRET")
SPARKAI_API_KEY = os.environ.get("SPARKAI_API_KEY")
config = Config(SPARKAI_APP_ID, SPARKAI_API_KEY, SPARKAI_API_SECRET)
if __name__ == "__main__":
    query = "成都有什么好玩的？"
    output_format = "" # User can specify output format
    profile = "" # User can define the role for LLM

    # Fetch web content based on the query
    web_contents_fetcher = WebContentFetcher(query)
    web_contents, serper_response = web_contents_fetcher.fetch()

    # Retrieve relevant documents using embeddings
    retriever = EmbeddingRetriever(config)
    relevant_docs_list = retriever.retrieve_embeddings(web_contents, serper_response['links'], query)
    print(relevant_docs_list)


    model_input = f'你是一个旅游攻略小助手，你的任务是，根据收集到的信息：\n{relevant_docs_list}.\n来精准回答用户所提出的问题：{query}。'
        #print(reranked)

    model = ChatModel(config, stream=False)
    output = model.generate([ChatMessage(role="user", content=model_input)])

    # print(output)
    # asd