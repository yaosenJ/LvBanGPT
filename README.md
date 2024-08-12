# “LvBan恣行” -- AI旅游助手

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
