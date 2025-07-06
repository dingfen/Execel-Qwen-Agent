# 新建agent.py文件，用于智能助手相关功能
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print
from qwen_server.user import UPLOAD_FOLDER, get_user_by_id
import re
import os

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
    system = (f'你现在是一位专业的 excel 处理助手，负责帮助用户处理 {UPLOAD_FOLDER} 内的 excel 文件，请尽可能使用python来完成用户的请求。但请注意，请务必在回答的最后加上对用户的追问，询问用户是否立即执行你生成的代码。')

    # 步骤2：定义您的工具
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
        'calculator'
    ]

    bot = Assistant(llm=llm_cfg,
                    function_list=tools,
                    system_message=system,)
    return bot

# 目前支持 @file:中文文件名.csv 的格式文件替换，后面必须有空格
# 支持 @sheet:sheet_name 的格式文件替换，后面必须有空格
# 支持 @range:A1:B2 的格式文件替换，后面必须有空格
def preprocess(user_id, raw_query):

    user = get_user_by_id(user_id)
    file_list = user.get_upload_files()
    pattern = r'@file:([\u4e00-\u9fa5a-zA-Z0-9_\-()\[\]{}., ]+\.(?:csv|xlsx|xls))\b'

    def replace_func(match):
        filename = match.group(1)
        full_path = os.path.join(UPLOAD_FOLDER, user_id, "excel", filename)
        return full_path

    query = re.sub(pattern, replace_func, raw_query)
    return query


# 用于流式响应的函数
def generate_response(bot, messages):
    from fastapi import HTTPException
    try:
        for response in bot.run(messages=messages):
            chunk = typewriter_print(response, '')
            yield f"data: {chunk}\n\n"  # SSE 格式
    except Exception as e:
        yield f"data: [Error] {str(e)}\n\n"