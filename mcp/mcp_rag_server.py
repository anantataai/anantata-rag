"""
MCP сервер для інтеграції RAG бази з Claude Desktop
"""
import asyncio
import json
import sys
import os
from typing import Optional

# ВАЖЛИВО: Додаємо шлях до server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from chatgpt_to_qdrant import ChatGPTToQdrant

# Ініціалізація RAG
rag_loader = ChatGPTToQdrant(qdrant_url="http://46.62.204.28:6333")

app = Server("rag-memory")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_memory",
            description="Пошук у RAG пам'яті ChatGPT/Claude",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Пошуковий запит"},
                    "top_k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10}
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_memory":
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        
        try:
            results = rag_loader.search(query=query, top_k=top_k)
            
            if not results:
                return [TextContent(type="text", text=f"❌ Нічого не знайдено для '{query}'")]
            
            text = "\n\n".join([
                f"{i+1}. {r.payload['text'][:300]}..." 
                for i, r in enumerate(results)
            ])
            return [TextContent(type="text", text=text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Помилка: {str(e)}")]
    
    return [TextContent(type="text", text=f"Невідомий інструмент: {name}")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
