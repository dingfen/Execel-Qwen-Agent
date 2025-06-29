from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import argparse
import uvicorn
import os
import base64


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

UPLOAD_FOLDER = '/mnt/h/tmp/excel/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

origins = ["http://0.0.0.0:8080", "http://localhost:8080"]  # 允许的跨域来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许跨域的源
    allow_credentials=True,  # 是否允许携带cookie
    allow_methods=["*"],  # 允许的方法，这里设置为所有方法
    allow_headers=["*"],  # 允许的请求头，这里设置为所有请求头
)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 上传文件
@app.post("/excel/upload")
async def upload_excel(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # 检查文件名是否为空
    if file.filename == '':
        raise HTTPException(status_code=400, detail="Empty filename")

    # 检查文件类型是否允许
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    # 限制文件大小（5MB）
    try:
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB")
        encoded_contents = base64.b64encode(contents).decode('utf-8')
    finally:
        await file.seek(0)

    # 保存文件到服务器
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(contents)
    

    return {
        "code": "success",
        "message": f"File {file.filename} uploaded successfully.",
        "filename": file.filename,
        "content": encoded_contents,
        "size": len(contents)
    }


@app.get("/excel/download")
def download_file(filename: str = Query(...)):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "rb") as f:
        content = f.read()
    return {"code": "success", "content": base64.b64encode(content).decode("utf-8")}

@app.get("/files/list")
def list_excel_files():
    try:
        # 获取指定目录下的所有文件
        files = os.listdir(UPLOAD_FOLDER)
        
        # 可选：过滤只显示 Excel 支持的文件类型
        excel_files = [f for f in files if allowed_file(f)]
        
        return {"code": "success", "files": excel_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Agent 响应 query
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