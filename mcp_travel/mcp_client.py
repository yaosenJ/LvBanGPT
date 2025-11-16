import asyncio
import json
import openai
from fastmcp import Client
import traceback


class MCPClient:
    def __init__(self):
        """
        初始化 MCP 客户端。
        """
        self.llm_client = openai.OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=" ",
        )
        self.model = "qwen-plus-latest"
        self.client = None
        self.tools = []
        self.servers = {}  # 存储多个服务器信息

    async def connect_to_multiple_mcp_servers(self, server_configs: dict):
        """
        连接到多个 MCP 服务器
        
        :param server_configs: 服务器配置字典
        {
            "weather_forecast_server": {
                "url": "http://127.0.0.1:8001/query_weather_mcp",
                "transport": "streamable-http"
            },
            "web_search_server": {
                "url": "http://127.0.0.1:8000/web_search_mcp",
                "transport": "streamable-http"
            },
            "search_nearby_server": {
                "url": "http://127.0.0.1:8002/search_nearby_mcp",
                "transport": "streamable-http"
            }
        }
        """
        try:
            # 配置多个 MCP 服务器
            config = {
                "mcpServers": {}
            }
            
            for server_name, server_info in server_configs.items():
                config["mcpServers"][server_name] = {
                    "url": server_info["url"],
                    "transport": server_info["transport"]
                }
                self.servers[server_name] = server_info["url"]
            
            # 创建 Client 实例
            self.client = Client(config)
            
            # 连接到服务器
            await self.client.__aenter__()
            
            # 获取所有可用工具
            self.tools = await self.client.list_tools()
            tool_names = [tool.name for tool in self.tools]
            
            print(f"成功连接到 {len(server_configs)} 个 MCP 服务器:")
            for server_name, server_url in self.servers.items():
                print(f"  - {server_name}: {server_url}")
            print(f"可用工具列表: {', '.join(tool_names)}")
            
            return True
            
        except Exception as e:
            print(f"连接失败: {e}")
            traceback.print_exc()
            return False

    async def connect_to_mcp_server_http(self, mcp_server_url: str):
        """
        连接到单个 MCP 服务器（保持向后兼容）
        """
        server_config = {
            "single_server": {
                "url": mcp_server_url,
                "transport": "streamable-http"
            }
        }
        return await self.connect_to_multiple_mcp_servers(server_config)

    def ping_llm_server(self):
        """
        向 LLM 服务器发送 ping 请求
        """
        messages = [{"role": "user", "content": "你是谁"}]
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            print(f"LLM服务器响应: {response.choices[0].message.content}")
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"LLM服务器连接错误: {str(e)}")
            return None

    async def call_tool(self, tool_name: str, arguments: dict):
        """调用工具的统一接口"""
        try:
            result = await self.client.call_tool(tool_name, arguments)
            return str(result)
        except Exception as e:
            return f"工具调用失败: {e}"

    def get_tool_descriptions(self):
        """获取所有工具的详细描述"""
        descriptions = []
        for tool in self.tools:
            desc = f"{tool.name}: {tool.description}"
            if hasattr(tool, 'inputSchema') and tool.inputSchema.get('properties'):
                params = tool.inputSchema['properties']
                param_desc = []
                for param_name, param_info in params.items():
                    param_desc.append(f"{param_name}({param_info.get('type', 'string')})")
                desc += f" [参数: {', '.join(param_desc)}]"
            descriptions.append(desc)
        return descriptions

    async def chat(self, query: str) -> str:
        """
        使用大模型处理查询并调用可用的 MCP 工具
        """
        messages = [{"role": "user", "content": query}]
        
        if not self.tools:
            # 如果没有可用工具，直接使用 LLM
            response = self.llm_client.chat.completions.create(
                model=self.model,            
                messages=messages
            )
            return response.choices[0].message.content
        
        # 转换为 OpenAI 工具格式
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": tool.inputSchema.get("type", "object"),
                    "properties": tool.inputSchema.get("properties", {}),
                    "required": tool.inputSchema.get("required", [])
                }
            }
        } for tool in self.tools]

        max_iterations = 5
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            try:
                # 调用 LLM
                response = self.llm_client.chat.completions.create(
                    model=self.model,            
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                )
                
                message = response.choices[0].message
                
                # 如果没有工具调用，直接返回回复
                if not message.tool_calls:
                    return message.content
                
                # 处理多个工具调用
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # 执行所有被调用的工具
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n[调用工具 {tool_name} 参数: {tool_args}]\n")
                    
                    # 调用工具
                    result = await self.call_tool(tool_name, tool_args)
                    print(f"[工具 {tool_name} 返回: {result}]\n")
                    
                    # 添加工具结果到消息历史
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    })
                
            except Exception as e:
                print(f"工具调用过程中出错: {e}")
                return f"处理过程中出现错误: {e}"
        
        # 达到最大迭代次数，返回最终回复
        final_response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return final_response.choices[0].message.content

    async def chat_loop(self):
        """
        命令行聊天循环
        """
        print("\nMCP 客户端运行中...")
        print("可用命令:")
        print("  - 'tools': 查看所有可用工具")
        print("  - 'servers': 查看连接的服务器")
        print("  - 'quit': 退出程序")
        print()

        while True:
            try:
                query = input("\n用户: ").strip()
                if query.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not query:
                    continue
                
                # 处理特殊命令
                if query.lower() == 'tools':
                    tool_descs = self.get_tool_descriptions()
                    print("\n可用工具:")
                    for desc in tool_descs:
                        print(f"  • {desc}")
                    continue
                elif query.lower() == 'servers':
                    print("\n已连接的服务器:")
                    for server_name, server_url in self.servers.items():
                        print(f"  • {server_name}: {server_url}")
                    continue
                    
                response = await self.chat(query)
                print(f"\nAI助手: {response}")

            except Exception as e:
                print(f"\n错误: {str(e)}")
                traceback.print_exc()

    async def cleanup(self):
        """
        清理资源
        """
        if self.client:
            await self.client.__aexit__(None, None, None)


async def main():
    """
    主函数
    """
    client = MCPClient()
    try:
        # 配置多个 MCP 服务器
        server_configs = {
            "weather_forecast_server": {
                "url": "http://127.0.0.1:8001/query_weather_mcp",
                "transport": "streamable-http"
            },
            "web_search_server": {
                "url": "http://127.0.0.1:8000/web_search_mcp",
                "transport": "streamable-http"
            },
            "search_nearby_server": {
                "url": "http://127.0.0.1:8002/search_nearby_mcp",
                "transport": "streamable-http"
            }
        }
        
        print("正在连接到多个 MCP 服务器...")
        success = await client.connect_to_multiple_mcp_servers(server_configs)
        if not success:
            print("连接失败，请检查服务器状态")
            return
        
        # 检查 LLM 服务器连接
        print("\n正在检查 LLM 服务器连接...")
        client.ping_llm_server()
        
        # 启动聊天循环
        await client.chat_loop()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
    finally:
        print("正在清理资源...")
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
