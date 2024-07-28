import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
import broadscope_bailian
import datetime
from http import HTTPStatus
from dashscope import Generation
import dashscope
dashscope.api_key ="sk-"
def llm(query,history=[],user_stop_words=[]):    # 调用api_server
    try:
        messages=[{'role':'system','content':'You are a helpful assistant.'}]
        for hist in history:
            messages.append({'role':'user','content':hist[0]})
            messages.append({'role':'assistant','content':hist[1]})
        messages.append({'role':'user','content':query})
        responses = Generation.call(
            model="qwen1.5-110b-chat",
            messages=messages,
            # 设置输出为'message'格式
            result_format='message',
            # 设置输出方式为流式输出
            stream=True,
            # 增量式流式输出
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
        
        # content=resp.get("Data", {}).get("Choices", [])[0].get("Message", {}).get("Content")
        #response = resp.get("Data").get("Text")
        return content
    except Exception as e:
        return str(e)
    
# travily搜索引擎
os.environ['TAVILY_API_KEY']='tvly-'    # travily搜索引擎api key
tavily=TavilySearchResults(max_results=10)
tavily.description='这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！'

# 工具列表
tools=[tavily, ]

tool_names='or'.join([tool.name for tool in tools])  # 拼接工具名
tool_descs=[] # 拼接工具详情
for t in tools:
    args_desc=[]
    for name,info in t.args.items():
        args_desc.append({'name':name,'description':info['description'] if 'description' in info else '','type':info['type']})
    args_desc=json.dumps(args_desc,ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s'%(t.name,t.description,args_desc))
tool_descs='\n'.join(tool_descs)

prompt_tpl='''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

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

def agent_execute(query,chat_history=[]):
    global tools,tool_names,tool_descs,prompt_tpl,llm,tokenizer
    
    agent_scratchpad='' # agent执行过程
    while True:
        # 1）触发llm思考下一步action
        history='\n'.join(['Question:%s\nAnswer:%s'%(his[0],his[1]) for his in chat_history])
        today=datetime.datetime.now().strftime('%Y-%m-%d')
        prompt=prompt_tpl.format(today=today,chat_history=history,tool_descs=tool_descs,tool_names=tool_names,query=query,agent_scratchpad=agent_scratchpad)
        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m'%prompt,flush=True)

        
        response=llm(prompt,user_stop_words=['Observation:'])
        print('\033[34m---LLM返回---\n%s\n---\033[34m'%response,flush=True)
        
        # 2）解析thought+action+action input+observation or thought+final answer
        thought_i=response.rfind('Thought:')
        final_answer_i=response.rfind('\nFinal Answer:')
        action_i=response.rfind('\nAction:')
        action_input_i=response.rfind('\nAction Input:')
        observation_i=response.rfind('\nObservation:')
        
        # 3）返回final answer，执行完成
        if final_answer_i!=-1 and thought_i<final_answer_i:
            final_answer=response[final_answer_i+len('\nFinal Answer:'):].strip()
            chat_history.append((query,final_answer))
            return True,final_answer,chat_history
        
        # 4）解析action
        if not (thought_i<action_i<action_input_i):
            return False,'LLM回复格式异常',chat_history
        if observation_i==-1:
            observation_i=len(response)
            response=response+'Observation: '
        thought=response[thought_i+len('Thought:'):action_i].strip()
        action=response[action_i+len('\nAction:'):action_input_i].strip()
        action_input=response[action_input_i+len('\nAction Input:'):observation_i].strip()
        
        # 5）匹配tool
        the_tool=None
        for t in tools:
            if t.name==action:
                the_tool=t
                break
        if the_tool is None:
            observation='the tool not exist'
            agent_scratchpad=agent_scratchpad+response+observation+'\n'
            continue 
        
        # 6）执行tool
        try:
            action_input=json.loads(action_input)
            tool_ret=the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation='the tool has error:{}'.format(e)
        else:
            observation=str(tool_ret)
        agent_scratchpad=agent_scratchpad+response+observation+'\n'

def agent_execute_with_retry(query,chat_history=[],retry_times=10):
    for i in range(retry_times):
        success,result,chat_history=agent_execute(query,chat_history=chat_history)
        if success:
            return success,result,chat_history
    return success,result,chat_history

my_history=[]
while True:
    query=input('query:')
    success,result,my_history=agent_execute_with_retry(query,chat_history=my_history)
    my_history=my_history[-10:]
