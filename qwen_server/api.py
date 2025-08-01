# 新建api.py文件，用于存放API接口定义
from fastapi import Depends, HTTPException, File, UploadFile, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from qwen_server.agent import whisper_model
import os
from pydub import AudioSegment

import base64
import os
import tempfile
import json
import subprocess
from qwen_agent.utils.output_beautify import typewriter_print
from qwen_server.user import verify_token, get_user_by_id, delete_user
from qwen_server.schema import QueryRequest

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = None):
    # token = credentials.credentials
    user_id = verify_token(None)
    user = get_user_by_id(user_id)
    if not user_id or not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user_id, user

# 用户相关API
def register_routes(app):
    @app.post("/user/register")
    def register_user(user_id: str, username: str = None):
        from qwen_server.user import create_user
        user = create_user(user_id, username)
        return {"message": "User registered successfully", "user": user.to_dict()}

    @app.get("/user/info")
    def get_user_info(credentials: HTTPAuthorizationCredentials = None):
        user_id, user = get_current_user(credentials)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"user": user.to_dict()}

    @app.delete("/user/delete")
    def delete_user_api(credentials: HTTPAuthorizationCredentials = None):
        user_id, _ = get_current_user(credentials)
        if delete_user(user_id):
            return {"message": "User deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="User not found")

    @app.get("/user/files")
    def list_user_files(credentials: HTTPAuthorizationCredentials = None):
        user_id, user = get_current_user(credentials)
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")
        return {"files": user.get_upload_files()}

    # 文件上传下载API
    @app.post("/excel/upload")
    @app.post("/excel/update")
    async def upload_excel(
        file: UploadFile = File(...),
        credentials: HTTPAuthorizationCredentials = None,
    ):
        user_id, user = get_current_user(credentials=credentials)
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # 检查文件名是否为空
        if file.filename == '':
            raise HTTPException(status_code=400, detail="Empty filename")

        # 检查文件类型是否允许
        if not user.allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="File type not allowed")

        # 限制文件大小（5MB）
        try:
            contents = await file.read()
            if len(contents) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size exceeds 5MB")
            encoded_contents = base64.b64encode(contents).decode('utf-8')
        finally:
            await file.seek(0)

        # 保存文件
        file_location = user.save_file(file.filename, contents)
        
        return {
            "code": "success",
            "message": f"File {file.filename} uploaded successfully.",
            "filename": file.filename,
            "content": encoded_contents,
            "size": len(contents)
        }

    @app.get("/excel/download")
    def download_file(
        filename: str = Query(...),
        credentials: HTTPAuthorizationCredentials = None,
    ):
        user_id, user = get_current_user(credentials)
        file_path = user.file_path(filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, "rb") as f:
            content = f.read()
        return {"code": "success", "content": base64.b64encode(content).decode("utf-8")}

    @app.get("/files/list")
    def list_excel_files(
        credentials: HTTPAuthorizationCredentials = None,
    ):
        user_id, user = get_current_user(credentials=credentials)
        files = user.get_upload_files()
        return {"code": "success", "files": files}

    # 查询API
    @app.post("/query")
    def process_query(request: QueryRequest, stream: bool = True, credentials: HTTPAuthorizationCredentials = None):
        from run_excel_mcp_server import excel_mcp_bot
        from qwen_server.agent import generate_response, preprocess
        
        user_id, user = get_current_user(credentials)
        query = preprocess(user_id, request.query)
        print(query)
        messages = [{'role': 'user', 'content': query}]

        if stream:
            return StreamingResponse(generate_response(excel_mcp_bot, messages), media_type="text/event-stream")
        else:
            try:
                response_plain_text = ''
                for response in excel_mcp_bot.run(messages=messages):
                    response_plain_text = typewriter_print(response, response_plain_text)
                return {"response": response_plain_text}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
    @app.post("/transcriptions")
    async def transcribe_audio(file: UploadFile = File(...)):
        # 检查文件类型
        if not file.filename.endswith(".webm"):
            raise HTTPException(status_code=400, detail="Only .webm files are allowed")

        # 保存上传的文件到临时路径
        temp_input_path = f"temp_{file.filename}"
        with open(temp_input_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        try:
            # 转换 .webm 到 .wav
            audio = AudioSegment.from_file(temp_input_path, format="webm")
            temp_wav_path = "temp_audio.wav"
            audio.export(temp_wav_path, format="wav")

            # 使用 Whisper 进行语音识别
            result = whisper_model.transcribe(temp_wav_path)
            transcription = result["text"]

            return {"transcription": transcription}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

    @app.post("/img2excel")
    async def img2excel(file: UploadFile = File(...), credentials: HTTPAuthorizationCredentials = None,):
        """
        处理表格图片，生成HTML和Excel文件并返回结果
        """
        user_id, user = get_current_user(credentials=credentials)
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # 检查文件名是否为空
        if file.filename == '':
            raise HTTPException(status_code=400, detail="Empty filename")

        # 检查文件类型是否允许
        if not user.allowed_image_file(file.filename):
            raise HTTPException(status_code=400, detail="File type not allowed")

        # 限制文件大小（5MB）
        try:
            contents = await file.read()
            if len(contents) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size exceeds 5MB")
            encoded_contents = base64.b64encode(contents).decode('utf-8')
        finally:
            await file.seek(0)


        file_location = user.save_file(file.filename, contents)

        try:
            # 处理表格图片 (这里整合 temp.py 的核心功能)
            # result = await image2excel(file_location, user.file_path(None))
            # 创建临时文件来存储参数和结果
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as params_file:
                params_file_path = params_file.name
                json.dump({
                    "input_image": file_location,
                    "output_dir": user.file_path(file.filename)
                }, params_file)
                
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as result_file:
                result_file_path = result_file.name
                
            # 使用 uvx 运行 image2excel 脚本
            # 假设你有一个名为 image2excel 的虚拟环境
            cmd = [
                "uvx",
                "--with-requirements", "ocr/requirements.txt",  # 替换为实际路径
                "python",
                "-c",
                f"""
import sys
import json
sys.path.append('ocr/')  # 替换为你的项目路径
from ocr.img2excel import image2excel
import asyncio

# 读取参数
with open('{params_file_path}', 'r') as f:
    params = json.load(f)

# 运行异步函数
result = asyncio.run(image2excel(params['input_image'], params['output_dir']))

# 保存结果
with open('{result_file_path}', 'w') as f:
    json.dump(result, f)
"""
            ]

            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            # 清理参数文件
            os.unlink(params_file_path)
            
            if result.returncode != 0:
                # 清理结果文件
                os.unlink(result_file_path)
                raise Exception(f"Image2Excel process failed: {result.stderr}")
                
            # 读取结果文件
            with open(result_file_path, 'r') as f:
                result_data = json.load(f)
            # 清理结果文件
            os.unlink(result_file_path)
            
            if "status" in result_data and result_data["status"] == "error":
                raise Exception(result_data["message"])
            
            response = {
                "status": "success",
                "filename": file.filename,
                "html_content": [],
                "excel_path": []
            }
            response["html_content"] = [ data['html_content'] for data in result_data.values() ]
            response["excel_path"] = [ data['excel_path'] for data in result_data.values() ]
            return response
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="Image processing timeout")
        except Exception as e:
            return {
                "status": "error",
                "message": f"处理失败: {str(e)}"
            }