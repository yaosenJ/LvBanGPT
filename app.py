import os
import gradio as gr
import requests
from gradio.components import HTML 
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
import FlagEmbedding 
import re
import time
import json
import numpy as np
from text2audio.infer import audio2lip
# 日志
from loguru import logger
from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
from http import HTTPStatus
from dashscope import Generation
import dashscope
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# 加载讯飞的api配置
SPARKAI_APP_ID = os.getenv("SPARKAI_APP_ID")
SPARKAI_API_SECRET = os.getenv("SPARKAI_API_SECRET")
SPARKAI_API_KEY = os.getenv("SPARKAI_API_KEY")
config = Config(SPARKAI_APP_ID, SPARKAI_API_KEY, SPARKAI_API_SECRET)
dashscope.api_key = os.getenv("dashscope_api_key")
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

# 第一阶段：用户上传图片并选择风格后，点击生成文案
def on_generate_click(image, style):
    generated_text = generate_text_from_image(image, style)
    return generated_text

# 第二阶段：点击“将文案转为语音”按钮，生成并播放语音
def on_convert_click(text_output):
    return text_to_audio(text_output)

# 第三阶段：点击“将文案转为数字人视频”按钮，生成并播放语音
def on_lip_click(text_output,video_path='./shuziren.mp4'):
    video_output = audio2lip(text_output,video_path)
    return video_output
    
rerank_path = '/mnt/workspace/LvBanGPT/model/rerank_model'
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
    #print(rerank_model_path)
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

# 获取城市信息 
def get_location_data(location,api_key):  
    """  
    向 QWeather API 发送 GET 请求以获取天气数据。  
  
    :param location: 地点名称或经纬度（例如："beijing" 或 "116.405285,39.904989"）  
    :param api_key: 你的 QWeather API 密钥  
    :return: 响应的 JSON 数据  
    """  
    # 构建请求 URL  
    url = f"https://geoapi.qweather.com/v2/city/lookup?location={location}&key={api_key}"  
  
    # 发送 GET 请求  
    response = requests.get(url)  
  
    # 检查响应状态码  
    if response.status_code == 200:  
        # 返回 JSON 数据  
        return response.json()
    else:  
        # 处理错误情况  
        print(f"请求失败，状态码：{response.status_code}")  
        print(response.text)  
        return None
    
# 获取天气  
def get_weather_forecast(location_id,api_key):  
    """  
    向QWeather API发送请求以获取未来几天的天气预报。  
  
    参数:  
    - location: 地点ID或经纬度  
    - api_key: 你的QWeather API密钥  
    - duration: 预报的时长，'3d' 或 '7d'  
  
    返回:  
    - 响应的JSON内容  
    """
    
    # 构建请求的URL  
    url = f"https://devapi.qweather.com/v7/weather/3d?location={location_id}&key={api_key}"  
  
    # 发送GET请求  
    response = requests.get(url)  
  
    # 检查请求是否成功  
    if response.status_code == 200:  
        # 返回响应的JSON内容  
        return response.json()  
    else:  
        # 如果请求不成功，打印错误信息  
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")  
        return None  
api_key = os.getenv("api_key")
from openai import OpenAI
client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
)

amap_key = os.getenv("amap_key")

def get_completion(messages, model="deepseek-chat"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        seed=1024,  # 随机种子保持不变，temperature 和 prompt 不变的情况下，输出就会不变
        tool_choice="auto",  # 默认值，由系统自动决定，返回function call还是返回文字回复
        tools=[{
            "type": "function",
            "function": {

                "name": "get_location_coordinate",
                "description": "根据POI名称，获得POI的经纬度坐标",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "POI名称，必须是中文",
                        },
                        "city": {
                            "type": "string",
                            "description": "POI所在的城市名，必须是中文",
                        }
                    },
                    "required": ["location", "city"],
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "search_nearby_pois",
                "description": "搜索给定坐标附近的poi",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "longitude": {
                            "type": "string",
                            "description": "中心点的经度",
                        },
                        "latitude": {
                            "type": "string",
                            "description": "中心点的纬度",
                        },
                        "keyword": {
                            "type": "string",
                            "description": "目标poi的关键字",
                        }
                    },
                    "required": ["longitude", "latitude", "keyword"],
                }
            }
        }],
    )
    return response.choices[0].message




def get_location_coordinate(location, city):
    url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print(url)
    r = requests.get(url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None


def search_nearby_pois(longitude, latitude, keyword):
    url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print(url)
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离：{distance}米\n\n"
    return ans
    

def process_request(prompt):
    messages = [
        {"role": "system", "content": "你是一个地图通，你可以找到任何地址。"},
        {"role": "user", "content": prompt}
    ]
    response = get_completion(messages)
    if (response.content is None):  # 解决 OpenAI 的一个 400 bug
        response.content = ""
    messages.append(response)  # 把大模型的回复加入到对话中
    print("=====GPT回复=====")
    print(response)
    
    # 如果返回的是函数调用结果，则打印出来
    while (response.tool_calls is not None):
        # 1106 版新模型支持一次返回多个函数调用请求
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(args)
    
            if (tool_call.function.name == "get_location_coordinate"):
                print("Call: get_location_coordinate")
                result = get_location_coordinate(**args)
            elif (tool_call.function.name == "search_nearby_pois"):
                print("Call: search_nearby_pois")
                result = search_nearby_pois(**args)
    
            print("=====函数返回=====")
            print(result)
    
            messages.append({
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)  # 数值result 必须转成字符串
            })
    
        response = get_completion(messages)
        if (response.content is None):  # 解决 OpenAI 的一个 400 bug
            response.content = ""
        messages.append(response)  # 把大模型的回复加入到对话中
    
    print("=====最终回复=====")
    print(response.content)
    return response.content

def llm(query, history=[], user_stop_words=[]):
    try:
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for hist in history:
            messages.append({'role': 'user', 'content': hist[0]})
            messages.append({'role': 'assistant', 'content': hist[1]})
        messages.append({'role': 'user', 'content': query})
        responses = Generation.call(
            model="qwen1.5-110b-chat",
            messages=messages,
            result_format='message',
            stream=True,
            incremental_output=True
        )
        content = ""
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                print(response)
                content += response.output.choices[0].message.content
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
        return content
    except Exception as e:
        return str(e)

# Travily 搜索引擎
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
tavily = TavilySearchResults(max_results=5)
tavily.description = '这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！'

# 工具列表
tools = [tavily]

tool_names = 'or'.join([tool.name for tool in tools])
tool_descs = []
for t in tools:
    args_desc = []
    for name, info in t.args.items():
        args_desc.append({'name': name, 'description': info['description'] if 'description' in info else '', 'type': info['type']})
    args_desc = json.dumps(args_desc, ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s' % (t.name, t.description, args_desc))
tool_descs = '\n'.join(tool_descs)

prompt_tpl = '''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

def agent_execute(query, chat_history=[]):
    global tools, tool_names, tool_descs, prompt_tpl, llm, tokenizer
    
    agent_scratchpad = ''  # agent执行过程
    while True:
        history = '\n'.join(['Question:%s\nAnswer:%s' % (his[0], his[1]) for his in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(today=today, chat_history=history, tool_descs=tool_descs, tool_names=tool_names, query=query, agent_scratchpad=agent_scratchpad)
        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m' % prompt, flush=True)

        response = llm(prompt, user_stop_words=['Observation:'])
        print('\033[34m---LLM返回---\n%s\n---\033[34m' % response, flush=True)
        
        thought_i = response.rfind('Thought:')
        final_answer_i = response.rfind('\nFinal Answer:')
        action_i = response.rfind('\nAction:')
        action_input_i = response.rfind('\nAction Input:')
        observation_i = response.rfind('\nObservation:')
        
        if final_answer_i != -1 and thought_i < final_answer_i:
            final_answer = response[final_answer_i + len('\nFinal Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history
        
        if not (thought_i < action_i < action_input_i):
            return False, 'LLM回复格式异常', chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response + 'Observation: '
        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()
        
        the_tool = None
        for t in tools:
            if t.name == action:
                the_tool = t
                break
        if the_tool is None:
            observation = 'the tool not exist'
            agent_scratchpad = agent_scratchpad + response + observation + '\n'
            continue 
        
        try:
            action_input = json.loads(action_input)
            tool_ret = the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation = 'the tool has error:{}'.format(e)
        else:
            observation = str(tool_ret)
        agent_scratchpad = agent_scratchpad + response + observation + '\n'

def agent_execute_with_retry(query, chat_history=[], retry_times=10):
    for i in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history

def process_network(query):
    my_history = []
    success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
    return result

# 旅行规划师功能

prompt = """你现在是一位专业的旅行规划师，你的责任是根据旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，帮助我规划旅游行程并生成详细的旅行计划表。请你以表格的方式呈现结果。旅行计划表的表头请包含日期、地点、行程计划、交通方式、餐饮安排、住宿安排、费用估算、备注。所有表头都为必填项，请加深思考过程，严格遵守以下规则：

1. 日期请以DayN为格式如Day1，明确标识每天的行程。
2. 地点需要呈现当天所在城市，请根据日期、考虑地点的地理位置远近，严格且合理制定地点，确保行程顺畅。
3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序。位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多。时间需要按照上午、中午、晚上制定，并给出每一个位置所停留的时间（如上午10点-中午12点）。活动需要准确描述在位置发生的对应活动（如参观博物馆、游览公园、吃饭等），并需根据位置停留时间合理安排活动类型。
4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择，如步行、地铁、出租车、火车、飞机等不同的交通方式，并尽可能详细说明。
5. 餐饮安排需包含每餐的推荐餐厅、类型（如本地特色、快餐等）、预算范围，就近选择。
6. 住宿安排需包含每晚的推荐酒店或住宿类型（如酒店、民宿等）、地址、预估费用，就近选择。
7. 费用估算需包含每天的预估总费用，并注明各项费用的细分（如交通费、餐饮费、门票费等）。
8. 备注中需要包括对应行程计划需要考虑到的注意事项，保持多样性，涉及饮食、文化、天气、语言等方面的提醒。
9. 请特别考虑随行人数的信息，确保行程和住宿安排能满足所有随行人员的需求。
10.旅游总体费用不能超过预算。

现在请你严格遵守以上规则，根据我的旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，生成合理且详细的旅行计划表。记住你要根据我提供的旅行目的地、天数等信息以表格形式生成旅行计划表，最终答案一定是表格形式。以下是旅行的基本信息：
旅游出发地：{}，旅游目的地：{} ，天数：{}天 ，行程风格：{} ，预算：{}，随行人数：{}, 特殊偏好、要求：{}

"""
def chat(chat_destination, chat_history, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other):
    stream_model = ChatModel(config, stream=True)
    final_query = prompt.format(chat_departure, chat_destination, chat_days, chat_style, chat_budget,  chat_people, chat_other)
    prompts = [ChatMessage(role='user', content=final_query)]
    # 将问题设为历史对话
    chat_history.append((chat_destination, ''))
    # 对话同时流式返回
    for chunk_text in stream_model.generate_stream(prompts):
        # 总结答案
        answer = chat_history[-1][1] + chunk_text
        # 替换最新的对话内容
        information = '旅游出发地：{}，旅游目的地：{} ，天数：{} ，行程风格：{} ，预算：{}，随行人数：{}'.format(chat_departure, chat_destination, chat_days, chat_style, chat_budget,  chat_people)
        chat_history[-1] = (information, answer)
        # 返回
        yield '', chat_history

# Gradio接口定义
with gr.Blocks() as demo:
    html_code = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 10px;
                }
                .container {
                    max-width: 1500px;
                    margin: auto;
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 10px;
                }
                .logo img {
                    display: block;
                    margin: 0 auto;
                    border-radius: 7px;
                }
                .content h2 {
                    text-align: center;
                    color: #999999;
                    font-size: 24px;
                    margin-top: 20px;
                }
                .content p {
                    text-align: center;
                    color: #cccccc;
                    font-size: 16px;
                    line-height: 1.5;
                    margin-top: 30px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">
                    <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/logo.png?raw=true" alt="Logo" width="30%">
                </div>
                <div class="content">
                    <h2>😀 亲爱的旅游爱好者们，欢迎来到“LvBan恣行”，您的专属旅行伙伴！我们致力于为您提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。</h2>
                <div class="hint" style="text-align: center;background-color: rgba(255, 255, 0, 0.15); padding: 10px; margin: 10px; border-radius: 5px; border: 1px solid #ffcc00;">
                    <p>“LvBan恣行”基于星火大模型的文生文、图生文以及文生语音等技术，旨在为您量身定制一份满意的旅行计划。无论您期望体验何种旅行目的地、天数、行程风格（如紧凑、适中或休闲）、预算以及随行人数，我们的助手都能为您精心规划行程并生成详尽的旅行计划表，包括每天的行程安排、交通方式以及需要注意的事项。</p>
                    <p>此外，我们还采用RAG技术，专为提供实用全方位信息而设计，包括景点推荐、活动安排、餐饮、住宿、购物、行程推荐以及实用小贴士等。目前，我们的知识库已涵盖全国各地区、城市的旅游攻略信息，为您提供丰富多样的旅行建议。</p>
                    <p>您还可以随时拍摄旅途中的照片，并通过我们的应用上传。应用将自动为您生成适应不同社交媒体平台（如朋友圈、小红书、抖音、微博）的文案风格，让您轻松分享旅途中的点滴，与朋友们共同感受旅游的乐趣。</p>
                    <p>立即加入“LvBan恣行”，让我们为您的旅行保驾护航，共同打造一段难忘的旅程！</p>
                </div>
                </div>
            </div>
        </body>
        </html>
"""
    gr.HTML(html_code)
    with gr.Tab("旅行规划助手"):
         # 输入框
        chat_departure = gr.Textbox(label="输入旅游出发地", placeholder="请你输入出发地")
        chat_destination = gr.Textbox(label="输入旅游目的地", placeholder="请你输入想去的地方")
        
        with gr.Accordion("个性化选择（天数，行程风格，预算，随行人数）", open=False):
            chat_days = gr.Slider(minimum=1, maximum=10, step=1, value=3, label='旅游天数')
            chat_style = gr.Radio(choices=['紧凑', '适中', '休闲'], value='适中', label='行程风格')
            chat_budget = gr.Textbox(label="输入预算(带上单位)", placeholder="请你输入预算")
            chat_people = gr.Textbox(label="输入随行人数", placeholder="请你输入随行人数")
            chat_other = gr.Textbox(label="特殊偏好、要求(可写无)", placeholder="请你特殊偏好、要求")
        # 聊天对话框
        chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天窗口", height=1000)
        # 按钮
        llm_submit_tab = gr.Button("发送", visible=True)
        # 问题样例
        gr.Examples(["合肥", "郑州", "西安", "北京", "广州", "大连"], chat_departure)
        gr.Examples(["北京", "南京", "大理", "上海", "东京", "巴黎"], chat_destination)
        # 按钮出发逻辑
        llm_submit_tab.click(fn=chat, inputs=[chat_destination, chatbot, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other], outputs=[ chat_destination,chatbot])
        
    with gr.Tab("旅游问答助手"):
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
        weather_input = gr.Textbox(label="请输入城市名查询天气", placeholder="例如：北京")
        weather_output = gr.HTML(value="", label="天气查询结果")
        query_button = gr.Button("查询天气")
        query_near = gr.Textbox(label="搜索附近的餐饮、酒店等", placeholder="例如：合肥市高新区中国声谷产业园附近的美食")
        result = gr.Textbox(label="查询结果", lines=10)
        submit_btn = gr.Button("查询附近的餐饮、酒店等")
        gr.Examples(["合肥市高新区中国声谷产业园附近的美食", "北京三里屯附近的咖啡", "南京市玄武区新街口附近的甜品店", "上海浦东新区陆家嘴附近的热门餐厅", "武汉市光谷步行街附近的火锅店", "广州市天河区珠江新城附近的酒店"], query_near)
        submit_btn.click(process_request, inputs=[query_near], outputs=[result])
        
        query_network = gr.Textbox(label="联网搜索问题", placeholder="例如：秦始皇兵马俑开放时间")
        result_network = gr.Textbox(label="搜索结果", lines=10)
        submit_btn_network = gr.Button("联网搜索")
        gr.Examples(["秦始皇兵马俑开放时间", "合肥有哪些美食", "北京故宫开放时间", "黄山景点介绍", "上海迪士尼门票需要多少钱"], query_network)
        submit_btn_network.click(process_network, inputs=[query_network], outputs=[result_network])
        
        Weather_APP_KEY = os.getenv("Weather_APP_KEY")
        def weather_process(location):
                api_key = Weather_APP_KEY  # 替换成你的API密钥  
                location_data = get_location_data(location, api_key)
                # print(location_data)
                if not location_data:
                    return "无法获取城市信息，请检查您的输入。"
                location_id = location_data.get('location', [{}])[0].get('id')
                # print(location_id)
                if not location_id:
                    return "无法从城市信息中获取ID。"
                weather_data = get_weather_forecast(location_id, api_key)
                if not weather_data or weather_data.get('code') != '200':
                    return "无法获取天气预报，请检查您的输入和API密钥。"
                # 构建HTML表格来展示天气数据
                html_content = "<table>"
                html_content += "<tr>"
                html_content += "<th>预报日期</th>"
                html_content += "<th>白天天气</th>"
                html_content += "<th>夜间天气</th>"
                html_content += "<th>最高温度</th>"
                html_content += "<th>最低温度</th>"
                html_content += "<th>白天风向</th>"
                html_content += "<th>白天风力等级</th>"
                html_content += "<th>白天风速</th>"
                html_content += "<th>夜间风向</th>"
                html_content += "<th>夜间风力等级</th>"
                html_content += "<th>夜间风速</th>"
                html_content += "<th>总降水量</th>"
                html_content += "<th>紫外线强度</th>"
                html_content += "<th>相对湿度</th>"
                html_content += "</tr>"

                for day in weather_data.get('daily', []):
                    html_content += f"<tr>"
                    html_content += f"<td>{day['fxDate']}</td>"
                    html_content += f"<td>{day['textDay']} ({day['iconDay']})</td>"
                    html_content += f"<td>{day['textNight']} ({day['iconNight']})</td>"
                    html_content += f"<td>{day['tempMax']}°C</td>"
                    html_content += f"<td>{day['tempMin']}°C</td>"
                    html_content += f"<td>{day.get('windDirDay', '未知')}</td>"
                    html_content += f"<td>{day.get('windScaleDay', '未知')}</td>"
                    html_content += f"<td>{day.get('windSpeedDay', '未知')} km/h</td>"
                    html_content += f"<td>{day.get('windDirNight', '未知')}</td>"
                    html_content += f"<td>{day.get('windScaleNight', '未知')}</td>"
                    html_content += f"<td>{day.get('windSpeedNight', '未知')} km/h</td>"
                    html_content += f"<td>{day.get('precip', '未知')} mm</td>"
                    html_content += f"<td>{day.get('uvIndex', '未知')}</td>"
                    html_content += f"<td>{day.get('humidity', '未知')}%</td>"
                    html_content += "</tr>"
                html_content += "</table>"  
  
                return HTML(html_content)  
        query_button.click(weather_process, [weather_input], [weather_output])
    

    with gr.Tab("旅行文案助手"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="上传图像")
            style_dropdown = gr.Dropdown(choices=style_options, label="选择风格模式", value="朋友圈")
            audio_output = gr.Audio(label="音频播放", interactive=False, visible=True)
            video_output = gr.Video(label="数字人",visible=True)

        with gr.Column():
            generate_button = gr.Button("生成文案", visible=True)
            generated_text = gr.Textbox(lines=8, label="生成的文案", visible=True)
                     
        generate_button.click(on_generate_click, inputs=[image_input, style_dropdown], outputs=[generated_text])
        convert_button1 = gr.Button("将文案转为语音", visible=True)
        convert_button1.click(on_convert_click, inputs=[generated_text], outputs=[audio_output])
        convert_button2 = gr.Button("将文案转为视频(请耐心等待)", visible=True)
        convert_button2.click(on_lip_click, inputs=[generated_text],outputs=[video_output])

# if __name__ == "__main__":
#     print("启动 Gradio 界面...")
#     demo.queue()  # 启用队列处理请求
#     demo.launch(root_path='/dsw-619620/proxy/7860/')

if __name__ == "__main__":
    demo.queue().launch(share=True)


