import datetime
import os
from typing import Tuple


# 核心节点类
class FilenameGenerator:
    # 节点分类（ComfyUI中显示的菜单路径）
    CATEGORY = "NNKK"
    # 输出类型：仅返回拼接后的字符串
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    # 核心执行函数
    FUNCTION = "filename_generator"

    # ========== 关键新增：禁用缓存 ==========
    # 禁用输出缓存（核心）
    CACHE_OUTPUTS = False
    # 禁用输入缓存（可选，增强版）
    CACHE_INPUTS = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制节点每次都被视为已更改"""
        return float("nan")  # 始终返回不同的值

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """定义三个输入参数：后缀字符串、前缀（时间格式）、保存目录"""
        return {
            "required": {
                # 注意：这里只定义参数格式，不解析时间，默认值改为纯提示文本
                "prefix": (
                    "STRING",
                    {
                        "default": "%Y%m%d_%H%M%S",
                        "multiline": False,
                        "placeholder": "时间格式掩码/固定字符串，如%Y%m%d、img_prefix等",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "固定字符串/时间掩码，如test、%H%M%S等",
                    },
                ),
            },
            "optional": {
                "save_directory": (
                    "STRING",
                    {
                        "default": "%Y-%m-%d",
                        "multiline": False,
                        "placeholder": "保存目录（支持时间掩码，如%Y%m%d、test_dir）",
                    },
                )
            },
        }

    def _format_with_current_time(self, input_str: str) -> str:
        """
        专用方法：使用**当前最新时间**解析时间掩码
        输入：可以是时间掩码（%Y%m%d）或普通字符串（test）
        输出：解析后的时间字符串 或 原字符串（非时间掩码时）
        """
        if not input_str:
            return ""

        # 每次调用都强制获取最新的系统时间（关键修复点）
        current_time = datetime.datetime.now()

        try:
            # 尝试用最新时间格式化输入字符串
            return current_time.strftime(input_str)
        except (ValueError, TypeError):
            # 不是合法时间掩码，返回原字符串
            return input_str

    def filename_generator(self, suffix: str, prefix: str, save_directory: str = "") -> Tuple[str]:
        """
        核心功能：
        1. 解析prefix/suffix/save_directory（都支持时间掩码）
        2. 拼接为 [目录\\]前缀_后缀 格式的文件名
        3. 每次执行都使用最新的系统时间
        """
        try:
            # 强制每次执行都重新解析所有参数（核心修复）
            # 即使参数输入框里的内容没改，也重新用当前时间解析
            parsed_prefix = self._format_with_current_time(prefix)
            parsed_suffix = self._format_with_current_time(suffix)

            # 拼接基础文件名，处理空值避免多余下划线
            file_parts = []
            if parsed_prefix:
                file_parts.append(parsed_prefix)
            if parsed_suffix:
                file_parts.append(parsed_suffix)
            file_name = "_".join(file_parts)

            # 处理保存目录（同样每次都重新解析）
            if save_directory:
                parsed_dir = self._format_with_current_time(save_directory)
                result = os.path.join(parsed_dir, file_name)
            else:
                result = file_name

            # 返回最终结果（元组格式，符合ComfyUI要求）
            return (result,)
        except Exception as e:
            # 异常处理：返回错误信息
            return (f"拼接失败：{str(e)}",)


# ComfyUI节点注册（必须）
NODE_CLASS_MAPPINGS = {
    "FilenameGeneratorNode": FilenameGenerator,
}

# 节点在界面上显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "FilenameGeneratorNode": "NNKK:FilenameGenerator",
}

