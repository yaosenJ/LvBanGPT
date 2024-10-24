<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/logo.png?raw=true" alt="image-20240131182121394" style="zoom:33%;" />

## 项目介绍

亲爱的旅游爱好者们，欢迎来到**LvBan恣行-AI旅游助手** ，您的专属旅行伙伴！我们致力于为您提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。

“LvBan恣行”基于**星火大模型**的文生文、图生文以及文生语音等技术，旨在为您量身定制一份满意的旅行计划。无论您期望体验何种旅行目的地、天数、行程风格（如紧凑、适中或休闲）、预算以及随行人数，我们的助手都能为您精心规划行程并生成详尽的旅行计划表，包括每天的行程安排、交通方式以及需要注意的事项等。

此外，我们还采用RAG技术，专为提供实用全方位信息而设计，包括景点推荐、活动安排、餐饮、住宿、购物、行程推荐以及实用小贴士等。目前，我们的知识库已涵盖全国各地区、城市的旅游攻略信息，为您提供丰富多样的旅行建议。

您还可以随时拍摄旅途中的照片，并通过我们的应用上传。应用将自动为您生成适应不同社交媒体平台（如朋友圈、小红书、抖音、微博）的文案风格，让您轻松分享旅途中的点滴，与朋友们共同感受旅游的乐趣。

立即加入“LvBan恣行”，让我们为您的旅行保驾护航，共同打造一段难忘的旅程！

 **功能模块**

- 旅游规划助手
- 旅游问答助手
- 旅行文案助手
  
**技术亮点**
- 充分使用**星火大模型API矩阵能力**，包含星火大模型、图片理解、语音合成、语音识别、文生图、embedding等
- 旅游规划、文案生成**Prompt**高效设计，**ReAct**提示框架设计
- **RAG**创新：根据用户query,**动态**加载，读取文本;**BM25检索**、**向量检索**的混合检索; **重排模型**高效使用
- 多模态生成：**图生文**，**文生图**，**TTS**，**ASR**和**数字人**视频合成
- 旅游问答**Agent**实现：**查询天气**、**附近搜索**、**联网搜索**
- 生成语音，生成图片和数字人视频全部可预览查看、下载，提高体验

## 项目整体功能逻辑流程图

<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/LvBan%E6%B5%81%E7%A8%8B%E5%9B%BE.png">
</p>

## 项目演示

- 旅游规划助手
  
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%97%85%E6%B8%B8%E8%A7%84%E5%88%92%E5%8A%A9%E6%89%8Bv2.0.gif" alt="Demo gif" >
</p>

- 旅游问答助手
  
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%97%85%E6%B8%B8%E9%97%AE%E7%AD%94%E5%8A%A9%E6%89%8Bv2.0.gif" alt="Demo gif" >
</p>

- 旅游文案助手
  
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%97%85%E6%B8%B8%E6%96%87%E6%A1%88%E5%8A%A9%E6%89%8Bv2.0.gif" alt="Demo gif" >
</p>

<details>
<summary>LvBan_v1.5项目展示</summary>
<br>

- 旅游规划助手
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%A7%84%E5%88%92%E5%8A%A9%E6%89%8B.gif" alt="Demo gif" >
</p>


- 旅游问答助手
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E9%97%AE%E7%AD%94%E5%8A%A9%E6%89%8B.gif" alt="Demo gif" >
</p>

- 旅游文案助手
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%96%87%E6%A1%88%E5%8A%A9%E6%89%8B.gif" alt="Demo gif" >
</p>
  
</details>



**开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐⭐ , 您的 star ⭐是我们最大的鼓励，谢谢各位！** 

## 🎉 NEWS
- [2024.09.26] **发布LvBan v2.0**演示视频：[https://www.bilibili.com/video/BV1wPxuepEBG](https://www.bilibili.com/video/BV1wPxuepEBG)
- [2024.09.24] 全新优化界面，**发布LvBan v2.0**
- [2024.09.20] 增加**语音识别对话**模块
- [2024.09.05] 增加**文生图**模块
- [2024.08.13] 项目介绍视频发布：[B站](https://www.bilibili.com/video/BV1pxYye6ECE)
- [2024.08.10] 完成PAI-DSW部署演示，以及操作文档撰写
- [2024.08.10] **发布LvBan v1.5**[modelscope](https://www.modelscope.cn/studios/NumberJys/LvBan)
- [2024.08.03] 进一步在旅行规划助手增加用户偏好设置（预算，随行人数）、特殊要求，帮助游客：**餐饮安排**、**住宿安排**、**费用估算**等
- [2024.07.31] 项目三个模块名称，分别改为**旅游规划助手**、**旅游问答助手**、**旅游文案助手**
- [2024.07.28] 改进的用户界面和用户体验（UI/UX）
- [2024.07.25] **重磅发布联网搜索、附近搜索Agent** 
- [2024.07.22] 增加**天气查询**功能
- [2024.07.21] **发布LvBan v1.0**[modelscope](https://www.modelscope.cn/studios/NumberJys/LvBan)
- [2024.07.20] **重磅发布 数字人 1.0** 🦸🦸🦸 
- [2024.07.19] **RAG优化**,增加混合检索技术（**BM25检索**、**向量检索**）、精排模型
- [2024.07.18] **接入 RAG 检索增强**，根据用户query,检索相应的pdf文本内容，高效准确地回答用户的旅游攻略问题
- [2024.07.16] 在文案生成模块，增加文本转语音**TTS**功能
- [2024.07.15] 旅游攻略数据收集、清洗
- [2024.07.14] 发布**旅游智能文案生成**应用,支持四种风格选择（**朋友圈**、**小红书**、**抖音**、**微博**），一键上传图片，生成对应风格文案
- [2024.07.11] 发布**旅游规划师**应用，根据旅游出发地、目的地、旅游天数以及旅游风格，形成旅行计划表，包括每天的行程安排、交通方式以及需要注意的事项 
- [2024.07.01] 借助星火大模型，开发简单**旅游问答**DEMO应用

## 🗂️ 目录

- [ 快速开始](#1-快速使用)
  - [在线体验](#11-在线体验)
  - [本地部署](#12-本地部署)
  - [PAI-DSW部署](#12-PAI-DSW部署)
- [详细指南](#2-详细指南)
  - [数据、模型及工具选型](#21-数据、模型及工具选型)
  - [基于本地旅游攻略pdf文本文件的RAG系统](#22-基于本地旅游攻略pdf文本文件的RAG系统)
  - [多模态生成：图生文，TTS和数字人视频合成](#23-多模态生成：图生文，TTS和数字人视频合成)
  - [旅游问答智能体(Agent)实现](#24-旅游问答智能体(Agent)实现)
- [案例展示](#3-案例展示)
- [人员贡献](#4-人员贡献)
- [ 致谢](#5-致谢)

<h2 id="1"> 快速使用 </h2>

<h3 id="1-1">在线体验 </h3>

- 目前已将 `LvBan v2.0` 版本部署到modelscope平台，地址: [https://www.modelscope.cn/studios/NumberJys/LvBan](https://www.modelscope.cn/studios/NumberJys/LvBan)
- `LvBan v2.0` 版本服务器部署链接：[http://120.27.156.91:1234/](http://120.27.156.91:1234/)  [https://1696-120-27-156-91.ngrok-free.app/](https://1696-120-27-156-91.ngrok-free.app/)  [https://872c-120-27-156-91.ngrok-free.app/](https://872c-120-27-156-91.ngrok-free.app/)
<h3 id="1-2">本地部署 </h3>

```bash
git clone https://github.com/yaosenJ/LvBanGPT.git
cd LvBanGPT
conda create -n LvBanGPT python=3.10.0 -y
conda activate  LvBanGPT
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
gradio app.py
```
<h3 id="1-3">PAI-DSW部署</h3>

#### 1. 登录ModelScope社区，选择`ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.14.0`魔搭GPU镜像，启动环境。地址: https://www.modelscope.cn/my/mynotebook/preset
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/PAI_DSW_GPU.png"  style="zoom:40%;" />
  
#### 2.克隆本项目仓库，环境依赖包安装

```bash
git clone https://github.com/yaosenJ/LvBanGPT.git
cd LvBanGPT
pip install -r requirements.txt
```
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E7%8E%AF%E5%A2%83%E5%8C%85%E4%B8%8B%E8%BD%BD.png"  style="zoom:40%;" />

#### 3.打开`down_rerank_model.ipynb`，下载重排模型

<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E9%87%8D%E6%8E%92%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD.png"  style="zoom:40%;" />
  
#### 4.`app.py`代码修改

<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E4%BF%AE%E6%94%B9%E4%BB%A3%E7%A0%81.png"  style="zoom:40%;" />

#### 5.修改.env文件，添加自己相关的key

vim .env 
```bash
SPARKAI_APP_ID=36f6
SPARKAI_API_SECRET=N2IzZDk0NjYzZDRjNmY3ZGUx
SPARKAI_API_KEY=a9b7b68d8bc752e79c4
x_api_key=c71505b1-9d4a-469a-b8d4-3
Weather_APP_KEY=417618eacc504fa4b7
amap_key=189a456127c050ee
api_key=sk-75ec0872c729491
dashscope_api_key=sk-064b2c7a65b
TAVILY_API_KEY=tvly-GH9Ma7CZyvmZK8Uq
```
#### 6.运行app.py,即python3 app.py,成功界面如下即可

<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%88%90%E5%8A%9F%E9%A1%B5%E9%9D%A2.png"  style="zoom:40%;" />

<h2 id="2"> 详细指南 </h2>

<h3 id="2-1"> 数据、模型及工具选型 </h3>

- 数据集：全国各地区及景点旅游攻略pdf文本文件
  
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%95%B0%E6%8D%AE.png"  style="zoom:40%;" />
  
- 大语言模型：星火大模型(Spark3.5 Max)
- 图片理解模型：星火图片理解模型
- 图片生成模型，星火文生图模型
- 语音合成模型：星火语音合成模型
- 语音识别模型：星火语音识别模型
- 向量模型：星火文本向量模型
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%AE%AF%E9%A3%9E%E5%BC%80%E6%94%BE%E5%B9%B3%E5%8F%B03.png" style="zoom:40%;" />
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E8%AE%AF%E9%A3%9E%E5%BC%80%E6%94%BE%E5%B9%B3%E5%8F%B02.png" style="zoom:40%;" />
  
<h3 id="2-2"> 基于本地旅游攻略pdf文本文件的RAG系统 </h3>
该项目的RAG系统，首先从用户的查询中提取关键信息（如城市名称、地点名称等），并通过这些信息检索匹配的pdf文件，提取相关内容并计算其嵌入向量。然后利用BM25检索和向量检索技术，筛选出与用户查询相似度较高的文本块。在此基础上，利用重排序模型对这些文本块进行进一步排序，最终选择最相关的内容提供给星火大模型。星火大模型根据这些上下文信息，生成对用户问题的准确回答。其详细技术实现流程：

### 1. 文本处理与文档匹配
   
- 城市提取（extract_cities_from_text）：使用jieba进行中文分词，提取query文本中提及的地名（城市名称）。
- PDF文件匹配（find_pdfs_with_city）：根据提取的城市名称，在指定目录下寻找包含这些城市名称的pdf文件。
  
### 2. 嵌入生成与文档处理
   
- PDF内容提取与分割（embedding_make）：
  - 1. 根据用户输入的文本，调用get_embedding_pdf函数提取相关的PDF文件。
  - 2. 从提取的pdf文件中读取文本内容，并对内容进行清理和分割，使用RecursiveCharacterTextSplitter将文本按(chunk_size=1000, chunk_overlap=300)进行切分，以便后续处理。
  - 3. 使用BM25Retriever对切分后的文本块进行初步检索，获得与用户问题最相关前20个文档片段。
     
### 3. 嵌入计算与相似度匹配
- 嵌入计算：
  - 1. 通过加载的EmbeddingModel(星火文本向量模型)，为用户的查询问题和检索到的文档片段生成嵌入向量。
  - 2. 使用余弦相似度（cosine_similarity）计算查询问题与文档片段之间的相似度。
  - 3. 根据相似度选择最相关的前10个文档片段。
     
### 4. 文档重排序与生成回答
   
- 重排序（rerank）：加载预训练的重排序模型(BAAI/bge-reranker-large)，对初步选出的文档片段进行进一步排序，选择出最相关的3个片段。
- 生成回答：
  - 1. 将重排序后的文档片段整合，并形成模型输入（通过指定的格式，将上下文和问题整合）。
  - 2. 调用ChatModel(星火大语言模型)生成最终回答，并返回给用户。

<h3 id="2-3"> 多模态生成：图生文，文生图，TTS，ASR和数字人视频合成 </h3>

<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E5%A4%9A%E6%A8%A1%E6%80%81%E7%94%9F%E6%88%90v2.0.png" style="zoom:40%;" />
  
通过将文本数据处理成音频数据后同视频一起输入，先使用脚本处理视频，该脚本首先会预先进行必要的预处理，例如人脸检测、人脸解析和 VAE 编码等。对音频和图片通过唇同步模型处理，生成对应唇形的照片，匹配所有的音频，最终将音频与生成的图片合成为视频输出。

<h3 id="2-4"> 旅游问答智能体(Agent)实现</h3>

- 查询天气Agent: 利用星火大模型（Spark3.5 Max）和 和风天气API实现联网搜索Agent。
- 附近搜索Agent: 利用星火大模型（Spark3.5 Max）和高德地图API实现附近搜索Agent。该Agent系统可以根据用户输入的文本请求，星火大模型自动判断是否需要调用高德地图API。若提问关于附近地址查询问题，则调用地图服务来获取地点信息和附近POI，目的帮助用户查询特定地点的周边设施、提供地址信息等，反之，其他问题，不调用高德地图API。
- 联网搜索Agent：利用星火大模型（Spark3.5 Max）和 Travily 搜索引擎API实现联网搜索Agent。

<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/Agent.png" style="zoom:40%;" />

### 4. 语音识别对话

运行asr.py即可
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/asr.png" style="zoom:40%;" />
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/asr_record.png" style="zoom:40%;" />


<h2 id="3"> 案例展示 </h2>

- 旅游规划助手
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E6%97%85%E6%B8%B8%E8%A7%84%E5%88%92v2.0.png" alt="Demo" >
- 知识库问答(RAG:true)
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E7%9F%A5%E8%AF%86%E5%BA%93%E9%97%AE%E7%AD%94v2.0_1.png" alt="Demo" >
- 知识库问答(RAG:false)
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E7%9F%A5%E8%AF%86%E5%BA%93%E9%97%AE%E7%AD%94v2.0_2.png" alt="Demo" >
- 附近查询&联网搜索&天气查询
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E5%AE%9E%E5%86%B5%E6%9F%A5%E8%AF%A2v2.0.png" alt="Demo" >
- 旅游文案助手
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E6%97%85%E6%B8%B8%E6%96%87%E6%A1%88v2.0.png" alt="Demo" >



<details>
<summary>LvBan_v1.5案例展示</summary>
<br>
<h2 id="3"> 案例展示 </h2>
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%97%85%E6%B8%B8%E6%94%BB%E7%95%A5.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/RAG.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E5%A4%A9%E6%B0%94%E6%9F%A5%E8%AF%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E9%99%84%E8%BF%91%E6%90%9C%E7%B4%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%81%94%E7%BD%91%E6%90%9C%E7%B4%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%96%87%E6%A1%88%E7%94%9F%E6%88%90.png" alt="Demo" >
</p>
</details>

<h2 id="4"> 人员贡献 </h2>

[yaosenJ](https://github.com/yaosenJ): 项目发起人，负责前后端开发

[shiqiyio](https://github.com/shiqiyio): 数字人、演示录制视频

[XiaoyangBi](https://github.com/XiaoyangBi): 产品功能规划，计划书撰写 

[qzd-1](https://github.com/qzd-1): 负责RAG模块

[Wwh-SoEximious](https://github.com/Wwh-SoEximious): 负责代码编写（天气查询功能），产品规划

[Dovislu](https://github.com/Dovislu): 数据收集处理，测试


<h2 id="5"> 致谢</h2>

感谢科大讯飞股份有限公司、共青团安徽省委员会、安徽省教育厅、安徽省科学技术厅和安徽省学生联合会联合举办的“**2024「星火杯」大模型应用创新赛**”！

感谢科大讯飞提供星火大模型API矩阵能力，包含星火大模型、星火语音识别大模型、图片理解、图片生成、超拟人语音合成、embedding等

感谢Datawhale及其教研团队在项目早期提供的基础教程

感谢Datawhale Amy大佬及其学习交流群的小伙伴们的支持和意见反馈！

感谢A100换你AD钙奶成员们的技术支持和反馈帮助！

感谢上海人工智能实验室，书生浦语大模型实战营的算力和计算平台支持！

## 参考资料

[星火大模型 python sdk库全面使用说明](https://github.com/iflytek/spark-ai-python)  

[星火大模型 python sdk库简易使用说明](https://pypi.org/project/dwspark/2024.0.2/)

[gradio前端展示案例](https://modelscope.cn/studios/Datawhale/datawhale_spark_2024)

[基于Assistant API的旅游助手的demo](https://help.aliyun.com/zh/model-studio/user-guide/assistant-api-based-travel-assistant?spm=a2c4g.11186623.0.0.1565c560IOXHpC)




