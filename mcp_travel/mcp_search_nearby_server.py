from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import requests

# 初始化MCP服务
mcp = FastMCP("POISearchServer")

# 假设已配置高德地图API密钥
amap_key = ""  # 请替换为实际API密钥

@mcp.tool(name="get_location_coordinate", 
          description="根据POI名称和所在城市获取经纬度坐标")
def get_location_coordinate(
    location: Annotated[str, Field(description="POI名称，必须为中文")],
    city: Annotated[str, Field(description="POI所在城市名，必须为中文")]
) -> str:
    """
    根据中文POI名称和所在城市，调用高德地图API获取其经纬度坐标
    
    Args:
        location: POI名称（中文）
        city: 所在城市（中文）
        
    Returns:
        包含经纬度的JSON字符串，格式如{"longitude": "xxx", "latitude": "xxx", "name": "xxx"}
        若未找到则返回空字符串
    """
    try:
        url = f"https://restapi.amap.com/v3/place/text?key={amap_key}&keywords={location}&region={city}"
        response = requests.get(url)
        result = response.json()
        
        if "pois" in result and result["pois"]:
            poi_info = result["pois"][0]
            return f'{{"longitude": "{poi_info["location"].split(",")[0]}", ' \
                   f'"latitude": "{poi_info["location"].split(",")[1]}", ' \
                   f'"name": "{poi_info["name"]}"}}'
        return ""
    except Exception as e:
        return f"获取坐标失败: {str(e)}"

@mcp.tool(name="search_nearby_pois", 
          description="根据经纬度坐标搜索附近的POI")
def search_nearby_pois(
    longitude: Annotated[str, Field(description="中心点经度")],
    latitude: Annotated[str, Field(description="中心点纬度")],
    keyword: Annotated[str, Field(description="目标POI的关键字")]
) -> str:
    """
    根据经纬度坐标和关键字，搜索周边的POI信息（最多返回3个结果）
    
    Args:
        longitude: 中心点经度
        latitude: 中心点纬度
        keyword: 搜索关键字（如"餐厅"、"加油站"等）
        
    Returns:
        格式化的POI信息，包含名称、地址和距离
    """
    try:
        url = f"https://restapi.amap.com/v3/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
        response = requests.get(url)
        result = response.json()
        
        ans = []
        if "pois" in result and result["pois"]:
            for i in range(min(3, len(result["pois"]))):
                poi = result["pois"][i]
                ans.append(
                    f"名称: {poi.get('name', '未知')}\n"
                    f"地址: {poi.get('address', '未知')}\n"
                    f"距离: {poi.get('distance', '未知')}米"
                )
        return "\n\n".join(ans) if ans else "未找到相关POI"
    except Exception as e:
        return f"搜索周边POI失败: {str(e)}"

# 启动服务
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8002,  path='/search_nearby_mcp')