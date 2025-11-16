import os
import json
from typing import List, Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field

# 1. Initialize MCP Server
mcp = FastMCP("SearchServer")
 
# 2. Setup Tavily Search
os.environ["TAVILY_API_KEY"] = " "
web_search_tool = TavilySearchResults(k=3)
 
def web_search(query: str) -> List[Dict[str, Any]]:
    """Perform web search for the given query."""
    if not query:
        return []
    print(f"正在搜索: {query}\n")
    results = web_search_tool.invoke({"query": query})
    # The tool returns a list of dicts, each with 'url' and 'content'
    # We need to transform it to a list of dicts with 'source' and 'content'
    return [{"source": r.get("url"), "content": r.get("content")} for r in results]
 
# 3. Formatting function
def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Formats search results into a readable string.
    :param results: A list of search result dictionaries.
    :return: A formatted string of search results.
    """
    if not results:
        return"No results found."
 
    formatted_string = "🔍 Here are the search results:\n\n"
    for i, result in enumerate(results):
        content = result.get('content', 'No Snippet')
        source = result.get('source', '#')
        # Use the first part of the content as a title, as Tavily doesn't provide one.
        title = content.split('\n')[0]
        formatted_string += f"{i+1}. **{title}**\n"
        formatted_string += f"   - Snippet: {content}\n"
        formatted_string += f"   - URL: {source}\n\n"
    return formatted_string.strip()

# 4. MCP Tool
@mcp.tool(name="web_search", description="search the web for information")
def search(query: Annotated[str, Field(description="The search query string - what you want to search for on the web")]) -> str:
    """
    Performs a web search for the given query and returns formatted results.
    
    Args:
        query: The search query string - what you want to search for on the web
        
    Returns:
        Formatted search results with titles, snippets and URLs
    """
    results = web_search(query)
    return format_search_results(results)

# 5. Run the server
if __name__ == "__main__":
    mcp.run(transport="streamable-http", path='/web_search_mcp')
