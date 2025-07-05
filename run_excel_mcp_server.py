from qwen_agent.agents import Assistant
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import argparse
import uvicorn
from qwen_server.api import register_routes
from qwen_server.agent import init_agent_service
    
app = FastAPI()

origins = ["http://0.0.0.0:8080", "http://localhost:8080"]  # 允许的跨域来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许跨域的源
    allow_credentials=True,  # 是否允许携带cookie
    allow_methods=["*"],  # 允许的方法，这里设置为所有方法
    allow_headers=["*"],  # 允许的请求头，这里设置为所有请求头
)

register_routes(app)
excel_mcp_bot = init_agent_service()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 FastAPI 服务并指定 host 和 port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定的主机地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=6006, help="绑定的端口号，默认 8000")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)