# 新建user.py文件，用于存放User类及相关操作
import os

UPLOAD_FOLDER = '/mnt/h/tmp/'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}


class User:
    def __init__(self, user_id: str, username: str = None, role: str = "user"):
        self.user_id = user_id
        self.username = username
        self.role = role  # 可以是 "admin", "user" 等
        self.upload_folder = os.path.join(self.get_upload_folder(), user_id, "excel")
        os.makedirs(self.upload_folder, exist_ok=True)

    @staticmethod
    def get_upload_folder():
        return UPLOAD_FOLDER

    def get_upload_files(self):
        """获取该用户上传的所有文件"""
        if not os.path.exists(self.upload_folder):
            return []
        files = os.listdir(self.upload_folder)
        return [f for f in files if self.allowed_file(f)]

    def file_path(self, filename: str):
        """返回该用户指定文件的路径"""
        return os.path.join(self.upload_folder, filename)

    def delete_file(self, filename: str):
        """删除用户指定文件"""
        path = self.file_path(filename)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def to_dict(self):
        """将用户信息转为字典格式用于 API 返回"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "file_count": len(self.get_upload_files())
        }

    def save_file(self, filename: str, content: bytes):
        """保存文件到用户目录"""
        file_path = self.file_path(filename)
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path

    @staticmethod
    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

test_user=User(user_id="test_user", username="test_user")

# 模拟用户数据库（实际应使用数据库）
users_db = {
    "test_user_id": test_user
}

def create_user(user_id: str, username: str = None):
    if user_id in users_db:
        return users_db[user_id]
    user = User(user_id=user_id, username=username)
    users_db[user_id] = user
    return user

def get_user_by_id(user_id: str):
    return users_db.get(user_id)

def delete_user(user_id: str):
    if user_id in users_db:
        del users_db[user_id]
        return True
    return False

def verify_token(token):
    # 这里应该实现实际的token验证逻辑
    # 现在我们只是模拟一个user_id返回
    return "test_user_id"