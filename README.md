<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/logo.png?raw=true" alt="image-20240131182121394" style="zoom:33%;" />

## 项目介绍

亲爱的旅游爱好者们，欢迎来到**LvBan恣行-AI旅游助手** ，您的专属旅行伙伴！我们致力于为您提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。

“LvBan恣行”基于**星火大模型**的文生文、图生文以及文生语音等技术，旨在为您量身定制一份满意的旅行计划。无论您期望体验何种旅行目的地、天数、行程风格（如紧凑、适中或休闲）、预算以及随行人数，我们的助手都能为您精心规划行程并生成详尽的旅行计划表，包括每天的行程安排、交通方式以及需要注意的事项等。

此外，我们还采用RAG技术，专为提供实用全方位信息而设计，包括景点推荐、活动安排、餐饮、住宿、购物、行程推荐以及实用小贴士等。目前，我们的知识库已涵盖全国各地区、城市的旅游攻略信息，为您提供丰富多样的旅行建议。

您还可以随时拍摄旅途中的照片，并通过我们的应用上传。应用将自动为您生成适应不同社交媒体平台（如朋友圈、小红书、抖音、微博）的文案风格，让您轻松分享旅途中的点滴，与朋友们共同感受旅游的乐趣。

立即加入“LvBan恣行”，让我们为您的旅行保驾护航，共同打造一段难忘的旅程！

 **功能模块**

- 根据天气查询结果，给外出旅游建议（连接天气预报API）
- 路径规划（飞机，火车）（相关购票app）
- 目的景点推荐（来自小红书等）
- 当地风俗礼仪，餐饮推荐（来自小红书等）

**参考资料**

星火大模型 python sdk库使用：

https://github.com/iflytek/spark-ai-python  
https://pypi.org/project/dwspark/2024.0.2/

gradio前端展示，案例：

https://modelscope.cn/studios/Datawhale/datawhale_spark_2024

基于Assistant API的旅游助手的demo：

https://help.aliyun.com/zh/model-studio/user-guide/assistant-api-based-travel-assistant?spm=a2c4g.11186623.0.0.1565c560IOXHpC


![image](https://github.com/yaosenJ/LvBanGPT/assets/147613954/f74be7b2-fc48-4c82-b903-c65c73f7b2ed)


当遇到上面问题，运行下面命令就可

```shell
git clone https://gitcode.net/miamnh/frpc_linux_amd64.git
mv frpc_linux_amd64 frpc_linux_amd64_v0.2
mv frpc_linux_amd64_v0.2 /root/.conda/lib/python3.11/site-packages/gradio
cd /root/.conda/lib/python3.11/site-packages/gradio
chmod +x frpc_linux_amd64_v0.2
cd app.py所在目录
gradio app.py
```

上传代码命令
```shell
 cd 对应项目一级目录/
 git add .
 git status(此命令选用，是用于查看状态的，可能显示一堆信息)
 git commit -m "随便写点啥"
 git push
```
