# 导入asyncio模块，用于支持异步操作
import asyncio
# 导入Client类，用于创建MCP客户端
from fastmcp import Client

# 配置MCP服务器连接信息
config = {
    "mcpServers": {
        "local-mcp-server": {
            "url": "http://127.0.0.1:8000/web_search_mcp",
            "transport": "streamable-http"
        }
    }
}

# 创建Client实例
client = Client(config)

# 定义异步函数call_tool，用于调用服务器上的工具
async def call_tool(name: str):
    """
    异步调用服务器上的hello工具
    
    参数:
        name: 字符串，要传递给hello工具的名称参数
    
    返回:
        无返回值，但会打印服务器的响应结果
    """
    # 使用异步上下文管理器连接到服务器
    async with client:
        # 调用服务器上的hello工具，传递name参数
        result = await client.call_tool("web_search", {"query": name})
        # 打印服务器返回的结果
        print(result)

# 运行异步函数call_tool，传入默认名称"Ford"
asyncio.run(call_tool("上海迪士尼乐园门票多少元一人？"))
