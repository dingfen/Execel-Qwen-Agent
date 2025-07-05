# 新建api.py文件，用于存放API接口定义
from fastapi import Depends, HTTPException, File, UploadFile, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import base64
import os

security = HTTPBearer()

# 请求模型
class QueryRequest(BaseModel):
    query: str

# 用户相关API
def register_routes(app):
    @app.post("/user/register")
    def register_user(user_id: str, username: str = None):
        from qwen_server.user import create_user
        user = create_user(user_id, username)
        return {"message": "User registered successfully", "user": user.to_dict()}

    @app.get("/user/info")
    def get_user_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
        from qwen_server.user import get_user_by_id, verify_token
        user_id = verify_token(credentials.credentials)
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"user": user.to_dict()}

    @app.delete("/user/delete")
    def delete_user_api(credentials: HTTPAuthorizationCredentials = Depends(security)):
        from qwen_server.user import delete_user, verify_token
        user_id = verify_token(credentials.credentials)
        if delete_user(user_id):
            return {"message": "User deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="User not found")

    @app.get("/user/files")
    def list_user_files(credentials: HTTPAuthorizationCredentials = Depends(security)):
        from qwen_server.user import get_user_by_id, verify_token
        user_id = verify_token(credentials.credentials)
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"files": user.get_upload_files()}

    # 文件上传下载API
    @app.post("/excel/upload")
    @app.post("/excel/update")
    async def upload_excel(
        file: UploadFile = File(...),
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        from qwen_server.user import get_user_by_id, verify_token, User
        
        user_id = verify_token(credentials.credentials)
        user = get_user_by_id(user_id)
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
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        from qwen_server.user import get_user_by_id, verify_token
        user_id = verify_token(credentials.credentials)
        user = get_user_by_id(user_id)
        file_path = user.file_path(filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, "rb") as f:
            content = f.read()
        return {"code": "success", "content": base64.b64encode(content).decode("utf-8")}

    @app.get("/files/list")
    def list_excel_files(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        from qwen_server.user import get_user_by_id, verify_token
        user_id = verify_token(credentials.credentials)
        user = get_user_by_id(user_id)
        files = user.get_upload_files()
        return {"code": "success", "files": files}

    # 查询API
    @app.post("/query")
    def process_query(request: QueryRequest, stream: bool = True):
        from qwen_server.agent import init_agent_service, generate_response
        bot = init_agent_service()
        messages = [{'role': 'user', 'content': request.query}]
        
        if stream:
            from fastapi import StreamingResponse
            return StreamingResponse(generate_response(bot, messages), media_type="text/event-stream")
        else:
            try:
                response_plain_text = ''
                for response in bot.run(messages=messages):
                    response_plain_text = typewriter_print(response, response_plain_text)
                return {"response": response_plain_text}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))