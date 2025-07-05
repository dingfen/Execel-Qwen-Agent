# 新建agent.py文件，用于智能助手相关功能
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import typewriter_print

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
    system = (f'你现在是一位专业的 excel 处理助手，负责帮助用户处理 {"/mnt/h/tmp/excel/"} 内的 excel 文件，请在合适的时机调用所需要的工具来帮助你完成用户的请求')

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

# 用于流式响应的函数
def generate_response(bot, messages):
    from fastapi import HTTPException
    try:
        for response in bot.run(messages=messages):
            chunk = typewriter_print(response, '')
            yield f"data: {chunk}\n\n"  # SSE 格式
    except Exception as e:
        yield f"data: [Error] {str(e)}\n\n"