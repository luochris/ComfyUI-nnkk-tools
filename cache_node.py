import os
import hashlib
import numpy as np
import torch
from PIL import Image

# 缓存目录配置
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_text")
os.makedirs(CACHE_DIR, exist_ok=True)


class LoadImagePrompt:
    """节点1：根据图片MD5读取缓存文本（新增skip_read开关）"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制节点每次都被视为已更改"""
        return float("nan")  # 始终返回不同的值

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI 原生图片输入（Tensor）
                "skip_load": ("BOOLEAN", {  # 新增：是否忽略读取的布尔参数
                    "default": False,  # 默认不忽略（正常读取）
                    "label_on": "skip_load",
                    "label_off": "normal_load"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "cache_key")
    FUNCTION = "load_text"
    CATEGORY = "NNKK"

    def load_text(self, image, skip_load):
        """
        核心逻辑：根据skip_load开关控制是否读取缓存
        :param image: 图片Tensor
        :param skip_load: bool，True=忽略读取（返回None），False=正常读取
        :return: 文本内容（None如果忽略/不存在）、图片MD5
        """
        # 1. 计算图片MD5（无论是否忽略读取，都计算MD5并输出）
        image_md5 = self.calculate_image_md5(image)

        # 2. 根据skip_load开关判断是否读取缓存
        if skip_load:
            # 忽略读取：直接返回None + MD5
            text_content = None
            print(f"🔍 已忽略读取缓存，图片MD5: {image_md5}")
        else:
            # 正常读取：拼接路径并读取文件
            cache_file_path = os.path.join(CACHE_DIR, f"{image_md5}.txt")
            if os.path.exists(cache_file_path):
                try:
                    with open(cache_file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    print(f"✅ 成功读取缓存: {cache_file_path}")
                except Exception as e:
                    print(f"❌ 读取缓存失败: {e}")
                    text_content = None
            else:
                text_content = None
                print(f"⚠️  缓存文件不存在: {cache_file_path}")

        return (text_content, image_md5)

    @staticmethod
    def calculate_image_md5(image):
        """
        增强版：适配所有ComfyUI图片格式，确保MD5计算一致
        """
        try:
            # 1. 统一处理Tensor/numpy
            if isinstance(image, torch.Tensor):
                # 处理空Tensor
                if image.nelement() == 0:
                    raise ValueError("图片Tensor为空")
                # 转移到CPU并转为numpy，强制float32避免精度问题
                image_np = image.detach().cpu().numpy().astype(np.float32)
            else:
                image_np = np.array(image).astype(np.float32)

            # 2. 强制标准化维度和数值范围
            # 压缩batch维度 [1, H, W, C] -> [H, W, C]
            image_np = image_np.squeeze(0)
            # 确保数值范围0~1（防止部分节点输出255范围的图片）
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
            # 强制转为uint8（固定精度，避免浮点误差）
            image_np = (image_np * 255).round().astype(np.uint8)

            # 3. 计算MD5（直接基于numpy数组字节，跳过PIL转换）
            md5_hash = hashlib.md5()
            # 强制按C顺序读取字节（避免不同系统/版本的字节序问题）
            md5_hash.update(image_np.tobytes(order='C'))
            return md5_hash.hexdigest()
        except Exception as e:
            print(f"计算图片MD5失败: {e}")
            raise


# 保存节点代码保持不变（此处省略，沿用之前的版本）
class SaveImagePrompt:
    """节点2：图片/手动缓存键二选一保存文本（均为可选，但必须选一个）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {  # 改为optional，让两个参数都可选
                "image": ("IMAGE",),  # 可选：图片（用于计算MD5）
                "cache_key": ("STRING", {  # 可选：手动输入的缓存键
                    "default": "",
                    "multiline": False
                }),
            },
            "required": {
                "prompt": ("STRING", {  # 必选：要保存的文本内容
                    "default": "",
                    "multiline": True  # 支持多行文本
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")  # 新增：输出实际使用的缓存键
    RETURN_NAMES = ("prompt", "cache_key")
    FUNCTION = "save_text"
    CATEGORY = "NNKK"

    def save_text(self, prompt, image=None, cache_key=None):
        """
        核心逻辑：图片和手动键二选一，保存文本到缓存
        :param prompt: 要保存的文本内容
        :param image: 可选：图片（用于计算MD5）
        :param cache_key: 可选：手动输入的缓存键
        :return: 保存的文本内容、实际使用的缓存键
        """
        # 1. 二选一判断：不能同时为空
        has_image = image is not None and image.nelement() > 0  # 检查Tensor是否非空
        has_manual_key = cache_key is not None and cache_key.strip() != ""

        if not has_image and not has_manual_key:
            raise ValueError("必须输入图片或缓存键中的一个！")

        # 2. 确定最终的缓存键（手动键优先，其次用图片MD5）
        if has_manual_key:
            final_cache_key = cache_key.strip()
        else:
            # 计算图片MD5
            final_cache_key = LoadImagePrompt.calculate_image_md5(image)

        # 3. 写入文本内容到缓存文件
        cache_file_path = os.path.join(CACHE_DIR, f"{final_cache_key}.txt")
        try:
            # 先检查目录是否可写
            if not os.access(os.path.dirname(cache_file_path), os.W_OK):
                raise PermissionError(f"没有写入权限: {os.path.dirname(cache_file_path)}")

            # 写入文件（添加flush确保立即写入）
            with open(cache_file_path, "w", encoding="utf-8", buffering=1) as f:
                f.write(prompt)
                f.flush()
            print(f"✅ 文本已保存到缓存: {cache_file_path}")
            print(f"🔑 使用的缓存键: {final_cache_key}")
            # 验证文件是否真的写入
            if os.path.getsize(cache_file_path) == 0:
                raise ValueError("保存的文件为空！")
        except PermissionError as e:
            print(f"❌ 权限错误: {e}")
            raise
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            raise

        # 输出：保存的内容 + 实际使用的缓存键（方便调试）
        return (prompt, cache_key)

NODE_CLASS_MAPPINGS = {
    "LoadImagePrompt": LoadImagePrompt,
    "SaveImagePrompt": SaveImagePrompt
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagePrompt": "NNKK:LoadImagePrompt",
    "SaveImagePrompt": "NNKK:SaveImagePrompt"
}