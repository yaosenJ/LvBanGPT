import os
import gradio as gr
import uuid
from sparkai.core.messages import ChatMessage,AIMessageChunk
from dwspark.config import Config
from dwspark.models import ChatModel, ImageUnderstanding, Text2Audio,Audio2Text,EmbeddingModel
from PIL import Image
import io
import base64
import random
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyMuPDFLoader
from sklearn.metrics.pairwise import cosine_similarity 
import gradio as gr
import pickle
import os
import re
import time
import numpy as np

from langchain_community.retrievers import BM25Retriever

# 日志
from loguru import logger
# 加载讯飞的api配置
SPARKAI_APP_ID = os.environ.get("SPARKAI_APP_ID")
SPARKAI_API_SECRET = os.environ.get("SPARKAI_API_SECRET")
SPARKAI_API_KEY = os.environ.get("SPARKAI_API_KEY")
config = Config(SPARKAI_APP_ID, SPARKAI_API_KEY, SPARKAI_API_SECRET)

# 初始化模型
iu = ImageUnderstanding(config)
t2a = Text2Audio(config)

# 临时存储目录
TEMP_IMAGE_DIR = "/tmp/sparkai_images/"
#AUDIO_TEMP_DIR = "/tmp/sparkai_audios/"

style_options = ["朋友圈", "小红书", "微博", "抖音"]

# 保存图片并获取临时路径
def save_and_get_temp_url(image):
    if not os.path.exists(TEMP_IMAGE_DIR):
        os.makedirs(TEMP_IMAGE_DIR)
    unique_filename = str(uuid.uuid4()) + ".png"
    temp_filepath = os.path.join(TEMP_IMAGE_DIR, unique_filename)
    image.save(temp_filepath)
    return temp_filepath

# 生成文本
def generate_text_from_image(image, style):
    temp_image_path = save_and_get_temp_url(image)
    prompt = "请理解这张图片"
    image_description = iu.understanding(prompt, temp_image_path)
    question = f"根据图片描述：{image_description}, 用{style}风格生成一段文字。"
    stream_model = ChatModel(config, stream=False)
    generated_text = stream_model.generate([ChatMessage(role="user", content=question)])
    return generated_text

# 文案到语音
def text_to_audio(text_input):
    try:
        audio_path = "./demo.mp3"
        t2a.gen_audio(text_input, audio_path)
        return audio_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return gr.Text.update(value="语音生成失败，请重试。")

# 第一阶段：用户上传图片并选择风格后，点击生成文案
def on_generate_click(image, style):
    generated_text = generate_text_from_image(image, style)
    return generated_text

# 第二阶段：点击“将文案转为语音”按钮，生成并播放语音
def on_convert_click(text_output):
    return text_to_audio(text_output)

rerank_path = './model/rerank_model'
rerank_model_name = 'BAAI/bge-reranker-large'
def extract_cities_from_text(text):
    # 从文本中提取城市名称，假设使用jieba进行分词和提取地名
    import jieba.posseg as pseg
    words = pseg.cut(text)
    cities = [word for word, flag in words if flag == "ns"]
    return cities

def find_pdfs_with_city(cities, pdf_directory):
    matched_pdfs = {}
    for city in cities:
        matched_pdfs[city] = []
        for root, _, files in os.walk(pdf_directory):
            for file in files:
                if file.endswith(".pdf") and city in file:
                    matched_pdfs[city].append(os.path.join(root, file))
    return matched_pdfs

def get_embedding_pdf(text, pdf_directory):
    # 从文本中提取城市名称
    cities = extract_cities_from_text(text)
    # 根据城市名称匹配PDF文件
    city_to_pdfs = find_pdfs_with_city(cities, pdf_directory)
    return city_to_pdfs


def load_rerank_model(model_name=rerank_model_name):
    """
    加载重排名模型。
    
    参数:
    - model_name (str): 模型的名称。默认为 'BAAI/bge-reranker-large'。
    
    返回:
    - FlagReranker 实例。
    
    异常:
    - ValueError: 如果模型名称不在批准的模型列表中。
    - Exception: 如果模型加载过程中发生任何其他错误。
    
    """ 
    if not os.path.exists(rerank_path):
        os.makedirs(rerank_path, exist_ok=True)
    rerank_model_path = os.path.join(rerank_path, model_name.split('/')[1] + '.pkl')
    # print(rerank_model_path)
    logger.info('Loading rerank model...')
    if os.path.exists(rerank_model_path):
        try:
            with open(rerank_model_path , 'rb') as f:
                reranker_model = pickle.load(f)
                logger.info('Rerank model loaded.')
                return reranker_model
        except Exception as e:
            logger.error(f'Failed to load embedding model from {rerank_model_path}') 
    else:
        try:
            os.system('apt install git')
            os.system('apt install git-lfs')
            os.system(f'git clone https://code.openxlab.org.cn/answer-qzd/bge_rerank.git {rerank_path}')
            os.system(f'cd {rerank_path} && git lfs pull')
    
            with open(rerank_model_path , 'rb') as f:
                reranker_model = pickle.load(f)
                logger.info('Rerank model loaded.')
                return reranker_model
                
        except Exception as e:
            logger.error(f'Failed to load rerank model: {e}')

def rerank(reranker, query, contexts, select_num):
        merge = [[query, context] for context in contexts]
        scores = reranker.compute_score(merge)
        sorted_indices = np.argsort(scores)[::-1]

        return [contexts[i] for i in sorted_indices[:select_num]]

def embedding_make(text_input, pdf_directory):

    city_to_pdfs = get_embedding_pdf(text_input, pdf_directory)
    city_list = []
    for city, pdfs in city_to_pdfs.items():
        print(f"City: {city}")
        for pdf in pdfs:
            city_list.append(pdf)
    
    if len(city_list) != 0:
        # all_pdf_pages = []
        all_text = ''
        for city in city_list:
            from pdf_read import FileOperation
            file_opr = FileOperation()
            try:
                text, error = file_opr.read(city)
            except:
                continue
            all_text += text
            
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        all_text = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), all_text)

        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300) 
        docs = text_spliter.create_documents([all_text])
        splits = text_spliter.split_documents(docs)
        question=text_input
        
        retriever = BM25Retriever.from_documents(splits)
        retriever.k = 20
        bm25_result = retriever.invoke(question)


        em = EmbeddingModel(config)
        question_vector = em.get_embedding(question)
        pdf_vector_list = []
        
        start_time = time.perf_counter()

        em = EmbeddingModel(config)  
        for i in range(len(bm25_result)):
            x = em.get_embedding(bm25_result[i].page_content) 
            pdf_vector_list.append(x)
            time.sleep(0.65)

        query_embedding = np.array(question_vector)
        query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, pdf_vector_list)

        top_k = 10
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

        emb_list = []
        for idx in top_k_indices:
            all_page = splits[idx].page_content
            emb_list.append(all_page)
        print(len(emb_list))

        reranker_model = load_rerank_model()

        documents = rerank(reranker_model, question, emb_list, 3)
        logger.info("After rerank...")
        reranked = []
        for doc in documents:
            reranked.append(doc)
        print(len(reranked))
        reranked = ''.join(reranked)

        model_input = f'你是一个旅游攻略小助手，你的任务是，根据收集到的信息：\n{reranked}.\n来精准回答用户所提出的问题：{question}。'
        #print(reranked)

        model = ChatModel(config, stream=False)
        output = model.generate([ChatMessage(role="user", content=model_input)])

        return output
    else:
        return "请在输入中提及想要咨询的城市！"

def process_question(history, use_knowledge_base, question, pdf_directory='./dataset'):
    if use_knowledge_base=='是':
        response = embedding_make(question, pdf_directory)
    else:
        model = ChatModel(config, stream=False)
        response = model.generate([ChatMessage(role="user", content=question)])
    
    history.append((question, response))
    return "", history

def clear_history(history):
    history.clear()
    return history
 

# 旅行规划师功能
prompt = '你现在是一位专业的旅行规划师，你的责任是根据旅行出发地，目的地、天数、行程风格（紧凑、适中、休闲），帮助我规划旅游行程并生成旅行计划表。请你以表格的方式呈现结果。 旅行计划表的表头请包含日期、地点、行程计划、交通方式、备注。所有表头都为必填项，请加深思考过程，严格遵守以下规则： 1. 日期请以DayN为格式如Day1。 2. 地点需要呈现当天所在城市，请根据日期、考虑地点的地理位置远近，严格且合理制定地点。 3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序，位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多，时间需要按照上午、中午、晚上制定并给出每一个位置所停留的时间如上午10点-中午12点，活动需要准确描述在位置发生的对应活动如参观xxx、游玩xxx、吃饭等，需根据位置停留时间合理安排活动类型。 4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择步行、地铁、飞机等不同的交通方式。 5. 备注中需要包括对应行程计划需要考虑到的注意事项，保持多样性。 现在请你严格遵守以上规则，根据我的旅行目的地、天数、行程风格（紧凑、适中、休闲），再以表格的方式生成合理的旅行计划表，提供表格后请再询问我行程风格、偏好、特殊要求等，并根据此信息完善和优化旅行计划表再次提供，直到我满意。记住你要根据我提供的旅行目的地、天数等信息以表格形式生成旅行计划表，最终答案一定是表格形式。旅游出发地：{}，旅游目的地：{} ，天数：{} ，行程风格：{}'

def chat(chat_destination, chat_history, chat_departure, chat_days, chat_style):
    stream_model = ChatModel(config, stream=True)
    final_query = prompt.format(chat_departure, chat_destination, chat_days, chat_style)
    prompts = [ChatMessage(role='user', content=final_query)]
    # 将问题设为历史对话
    chat_history.append((chat_destination, ''))
    # 对话同时流式返回
    for chunk_text in stream_model.generate_stream(prompts):
        # 总结答案
        answer = chat_history[-1][1] + chunk_text
        # 替换最新的对话内容
        chat_history[-1] = (chat_destination, answer)
        # 返回
        yield '', chat_history

# Gradio接口定义
with gr.Blocks() as demo:
    with gr.Tab("旅行规划师"):
        warning_html_code = """
                <div class="hint" style="text-align: center;background-color: rgba(255, 255, 0, 0.15); padding: 10px; margin: 10px; border-radius: 5px; border: 1px solid #ffcc00;">
                    <p>🐱 欢迎来到LvBan旅游助手，根据您提供的旅行出发地，目的地、天数、行程风格（紧凑、适中、休闲），帮助您规划旅游行程并生成旅行计划表</p>
                    <p>相关地址: <a href="https://challenge.xfyun.cn/h5/xinghuo?ch=dwm618">比赛地址</a>、<a href="https://github.com/yaosenJ/LvBanGPT">项目地址</a></p>
                </div>
                """
        gr.HTML(warning_html_code)
        # 输入框
        chat_departure = gr.Textbox(label="输入旅游出发地", placeholder="请你输入出发地")
        chat_destination = gr.Textbox(label="输入旅游目的地", placeholder="请你输入想去的地方")
        
        chat_days = gr.Radio(choices=['1天', '2天', '3天', '4天', '5天', '6天', '7天', '8天', '9天', '10天'], value='3天', label='旅游天数')
        chat_style = gr.Radio(choices=['紧凑', '适中', '休闲'], value='适中', label='行程风格')
        
        # 聊天对话框
        chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天历史")
        # 按钮
        llm_submit_tab = gr.Button("发送", visible=True)
        # 问题样例
        gr.Examples(["合肥", "郑州", "西安", "北京", "广州", "大连"], chat_departure)
        gr.Examples(["北京", "南京", "大理", "上海", "东京", "巴黎"], chat_destination)
        # 按钮出发逻辑
        llm_submit_tab.click(fn=chat, inputs=[chat_destination, chatbot, chat_departure, chat_days, chat_style], outputs=[chat_destination, chatbot])
        
    with gr.Tab("旅行攻略小卫士"):
        warning_html_code = """
            <div class="hint" style="text-align: center;background-color: rgba(255, 255, 0, 0.15); padding: 10px; margin: 10px; border-radius: 5px; border: 1px solid #ffcc00;">
                <p>🐱 欢迎来到LvBan旅游助手，我可以提供景点推荐、活动安排、餐饮、住宿、购物、行程推荐、实用小贴士等实用全方位信息</p>
                <p>目前知识库包含全国各地区、城市旅游攻略信息。如：大连、香港、贵阳、北京、黄山、新疆、厦门、丽江等几百个景点</p>
                <p>相关地址: <a href="https://challenge.xfyun.cn/h5/xinghuo?ch=dwm618">比赛地址</a>、<a href="https://github.com/yaosenJ/LvBanGPT">项目地址</a></p>
            </div>
            """
        gr.HTML(warning_html_code)
        chatbot = gr.Chatbot(label="聊天记录")
        msg = gr.Textbox(lines=2,placeholder="请输入您的问题（旅游景点、活动、餐饮、住宿、购物、推荐行程、小贴士等实用信息）",label="提供景点推荐、活动安排、餐饮、住宿、购物、行程推荐、实用小贴士等实用信息")
        whether_rag = gr.Radio(choices=['是','否'], value='否', label='是否启用RAG')
        submit_button = gr.Button("发送")
        clear_button = gr.Button("清除对话")
        # 问题样例
        gr.Examples(["我想去香港玩，你有什么推荐的吗？","我计划暑假带家人去云南旅游，请问有哪些必游的自然风光和民族文化景点？","下个月我将在西安，想了解秦始皇兵马俑开通时间以及交通信息","第一次去西藏旅游，需要注意哪些高原反应的预防措施？","去三亚度假，想要住海景酒店，性价比高的选择有哪些？","去澳门旅游的最佳时间是什么时候？","计划一次五天四夜的西安深度游，怎样安排行程比较合理，能覆盖主要景点？","在杭州，哪些家餐馆可以推荐去的？"], msg)
        def respond(message, chat_history, use_kb):
            return process_question(chat_history, use_kb, message)

        def clear_chat(chat_history):
            return clear_history(chat_history)

        submit_button.click(respond, [msg, chatbot, whether_rag], [msg, chatbot])
        clear_button.click(clear_chat, chatbot, chatbot)
    

    with gr.Tab("旅行智能文案生成"):
        warning_html_code = """
                <div class="hint" style="text-align: center;background-color: rgba(255, 255, 0, 0.15); padding: 10px; margin: 10px; border-radius: 5px; border: 1px solid #ffcc00;">
                    <p>🐱 欢迎来到LvBan旅游助手，根据你随手拍的照片，上传到该应用，自动生成你想要的文案风格模式（朋友圈、小红书、抖音、微博），然后分享给大家，一起享受旅游愉快。</p>
                    <p>相关地址: <a href="https://challenge.xfyun.cn/h5/xinghuo?ch=dwm618">比赛地址</a>、<a href="https://github.com/yaosenJ/LvBanGPT">项目地址</a></p>
                </div>
                """
        gr.HTML(warning_html_code)
        with gr.Row():
            image_input = gr.Image(type="pil", label="上传图像")
            style_dropdown = gr.Dropdown(choices=style_options, label="选择风格模式", value="朋友圈")
            audio_output = gr.Audio(label="音频播放", interactive=False, visible=True)

        with gr.Column():
            generate_button = gr.Button("生成文案", visible=True)
            generated_text = gr.Textbox(label="生成的文案", visible=True)

        generate_button.click(on_generate_click, inputs=[image_input, style_dropdown], outputs=[generated_text])
        convert_button = gr.Button("将文案转为语音", visible=True)
        convert_button.click(on_convert_click, inputs=[generated_text], outputs=[audio_output])



if __name__ == "__main__":
    demo.queue().launch(share=True)


