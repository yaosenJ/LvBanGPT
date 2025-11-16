from __future__ import annotations

import os
import json
import httpx
from typing import Any
from dotenv import load_dotenv
from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")

# OpenWeather API 配置
# OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = ""     # 填写你的OpenWeather-API-KEY
USER_AGENT = "weather-app/1.0"

# 替换API端点
OPENWEATHER_FORECAST_API = "https://api.openweathermap.org/data/2.5/forecast"

async def fetch_weather_forecast(city: str, days: int = 1) -> dict[str, Any] | None:
    """
    获取未来几天天气预报
    :param city: 城市名称
    :param days: 天数 (1-5)
    :return: 天气预报数据
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_FORECAST_API, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # 处理返回数据，按天数筛选
            return process_forecast_data(data, days)
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}

def process_forecast_data(data: dict, days: int) -> dict:
    """处理预报数据，按天数筛选"""
    # OpenWeather 5天预报每3小时一个数据点
    # 这里需要根据天数筛选相应的数据
    forecast_list = data.get("list", [])
    
    # 简单的实现：取前 days*8 个数据点（每天8个）
    points_to_show = min(days * 8, len(forecast_list))
    data["list"] = forecast_list[:points_to_show]
    
    return data

@mcp.tool(name="query_weather_forecast", description="查询未来几天天气预报")
async def query_weather_forecast(
    city: Annotated[str, Field(description="城市名称（需使用英文")],
    days: Annotated[int, Field(description="预报天数 (1-5)", ge=1, le=5)] = 1
) -> str:
    """
    查询未来几天天气预报
    :param city: 城市名称
    :param days: 预报天数 (1-5)
    :return: 格式化后的天气预报信息
    """
    data = await fetch_weather_forecast(city, days)
    return format_forecast(data, days)

def format_forecast(data: dict[str, Any], days: int) -> str:
    """格式化天气预报数据"""
    if "error" in data:
        return f"⚠️ {data['error']}"
    
    city = data.get("city", {}).get("name", "未知")
    country = data.get("city", {}).get("country", "未知")
    
    result = [f"🌍 {city}, {country} - 未来{days}天天气预报\n"]
    
    for i, forecast in enumerate(data.get("list", [])[:days*8]):
        dt_txt = forecast.get("dt_txt", "")
        temp = forecast.get("main", {}).get("temp", "N/A")
        description = forecast.get("weather", [{}])[0].get("description", "未知")
        
        result.append(f"📅 {dt_txt} | 🌡 {temp}°C | 🌤 {description}")
    
    return "\n".join(result)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001,  path='/query_weather_mcp')