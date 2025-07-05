from qwen_agent.agents import Assistant
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import argparse
import uvicorn
from qwen_server.api import register_routes
from qwen_server.user import UPLOAD_FOLDER

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
    system = (f'你现在是一位专业的 excel 处理助手，负责帮助用户处理 {UPLOAD_FOLDER} 内的 excel 文件，请在合适的时机调用所需要的工具来帮助你完成用户的请求')

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

    bot = Assistant(llm=llm_cfg,
                    function_list=tools,
                    system_message=system,)
    return bot
    
app = FastAPI()
bot = init_agent_service()

origins = ["http://0.0.0.0:8080", "http://localhost:8080"]  # 允许的跨域来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许跨域的源
    allow_credentials=True,  # 是否允许携带cookie
    allow_methods=["*"],  # 允许的方法，这里设置为所有方法
    allow_headers=["*"],  # 允许的请求头，这里设置为所有请求头
)

register_routes(app)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 FastAPI 服务并指定 host 和 port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定的主机地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=6006, help="绑定的端口号，默认 8000")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)