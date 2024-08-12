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
- 充分使用**星火大模型API矩阵能力**，包含星火大模型、图片理解、超拟人语音合成、embedding等
- 旅游规划、文案生成**Prompt**高效设计，**ReAct**提示框架设计
- **RAG**创新：根据用户query,**动态**加载，读取文本;**BM25检索**、**向量检索**的混合检索; **重排模型**高效使用
- 多模态生成：**图生文**，**TTS**和**数字人**视频合成
- 旅游问答**Agent**实现：**查询天气**、**附近搜索**、**联网搜索**
- tts和数字人视频全部可预览查看、下载，提高体验

## 项目整体功能逻辑流程图
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/LvBan%E6%B5%81%E7%A8%8B%E5%9B%BE.png" alt="image-20240131182121394" style="zoom:100%;" />

## 项目演示

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


**开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐⭐ , 您的 star ⭐是我们最大的鼓励，谢谢各位！** 

## 🎉 NEWS

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

目前已将 `LvBan v1.5` 版本部署到modelscope平台，地址: [https://www.modelscope.cn/studios/NumberJys/LvBan](https://www.modelscope.cn/studios/NumberJys/LvBan)

<h3 id="1-2">本地部署 </h3>

```shell

conda create -n LvBanGPT python=3.10.0 -y
conda activate  LvBanGPT
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
gradio app.py
```
<h3 id="1-3">PAI-DSW部署</h3>
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/PAI_DSW.png"  style="zoom:40%;" />

<h2 id="2"> 详细指南 </h2>

<h3 id="2-1"> 数据、模型及工具选型 </h3>

- 数据集：全国各地区及景点旅游攻略pdf文本文件
  
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%95%B0%E6%8D%AE.png"  style="zoom:40%;" />
  
- 大语言模型：星火大模型(Spark3.5 Max)
- 图片理解模型：星火图片理解模型
- 语音合成模型：星火语音合成模型
- 向量模型：星火文本向量模型
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%AE%AF%E9%A3%9E%E5%BC%80%E6%94%BE%E5%B9%B3%E5%8F%B0.png" style="zoom:40%;" />
  
<h3 id="2-2"> 基于本地旅游攻略pdf文本文件的RAG系统 </h3>


<h3 id="2-3"> 多模态生成：图生文，TTS和数字人视频合成 </h3>


<h3 id="2-4"> 旅游问答智能体(Agent)实现</h3>

<h2 id="3"> 案例展示 </h2>
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/RAG.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E5%A4%A9%E6%B0%94%E6%9F%A5%E8%AF%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E9%99%84%E8%BF%91%E6%90%9C%E7%B4%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%81%94%E7%BD%91%E6%90%9C%E7%B4%A2.png" alt="Demo" >
</p>

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




