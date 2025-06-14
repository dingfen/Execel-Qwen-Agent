from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import argparse
import uvicorn


def init_agent_service():
    llm_cfg = {
        'model': 'qwen3:8b',
        'model_server': 'http://localhost:11434/v1',  # Ollama API
        'api_key': 'EMPTY',
        'generate_cfg': {
            'extra_body': {
                'chat_template_kwargs': {'enable_thinking': False}
            }
        },
        'thought_in_content': True,
    }

    # 步骤2：定义您的工具（MCP + 代码解释器）
    tools = [
        {'mcpServers': {
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            'fetch': {
                'command': 'uvx',
                'args': ['mcp-server-fetch']
            }
        }},
        'code_interpreter',
        {
            "mcpServers": {
                "excel-stdio": {
                    "command": "uv",
                    "args": ["run", "excel-mcp-server", "stdio"]
                }
            }
        }
    ]

    bot = Assistant(llm=llm_cfg, function_list=tools)
    return bot
    
class QueryRequest(BaseModel):
    query: str


app = FastAPI()
bot = init_agent_service()


@app.post("/query")
def  process_query(request: QueryRequest, stream: bool = True):
    messages = [{'role': 'user', 'content': request.query}]
    response_plain_text = ''

    if stream:
        async def generate():
            nonlocal response_plain_text
            try:
                for response in bot.run(messages=messages):
                    chunk = typewriter_print(response, response_plain_text)
                    response_plain_text = chunk
                    yield f"data: {chunk}\n\n"  # SSE 格式
            except Exception as e:
                yield f"data: [Error] {str(e)}\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            for response in bot.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)
            return {"response": response_plain_text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 FastAPI 服务并指定 host 和 port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定的主机地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=6006, help="绑定的端口号，默认 8000")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)