import yaml
import os
from fetch_web_content import WebContentFetcher
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
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
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity 
import pickle
import re
import time
import numpy as np
# from text2audio.infer import audio2lip
# 日志
from loguru import logger
# 加载讯飞的api配置



class EmbeddingRetriever:
    def __init__(self, config):
        self.config = config
        # Initialize the text splitter
        self.text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300
        )

    def retrieve_embeddings(self, contents_list: list, link_list: list, query: str):
        # Retrieve embeddings for a given list of contents and a query
        metadatas = [{'url': link} for link in link_list]
        texts = self.text_spliter.create_documents(contents_list, metadatas=metadatas)
        splits = self.text_spliter.split_documents(texts)

        retriever = BM25Retriever.from_documents(splits)
        retriever.k = 5
        bm25_result = retriever.invoke(query)


        em = EmbeddingModel(self.config)
        query_vector = em.get_embedding(query)
        pdf_vector_list = []

        for i in range(len(bm25_result)):
            x = em.get_embedding(bm25_result[i].page_content) 
            pdf_vector_list.append(x)
            time.sleep(0.65)
        
        query_embedding = np.array(query_vector)
        query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, pdf_vector_list)

        top_k = 3
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

        emb_list = []
        for idx in top_k_indices:
            all_page = splits[idx].page_content
            all_page_list = all_page.split('\r\n')
            stripped_lines = [line.strip() for line in all_page_list]

            all_page = '\n'.join(stripped_lines)
            all_page = re.sub(r'\n+', '\n', all_page)
            emb_list.append(all_page)

        return emb_list # Retrieve and return the relevant documents


# Example usage
if __name__ == "__main__":
    query = "香港旅游攻略，美食攻略。"

    # Create a WebContentFetcher instance and fetch web contents
    web_contents_fetcher = WebContentFetcher(query)
    web_contents, serper_response = web_contents_fetcher.fetch()

    # Create an EmbeddingRetriever instance and retrieve relevant documents
    retriever = EmbeddingRetriever()
    relevant_docs_list = retriever.retrieve_embeddings(web_contents, serper_response['links'], query)

    print("\n\nRelevant Documents from VectorDB:\n", relevant_docs_list)
    