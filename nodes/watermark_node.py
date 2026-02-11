import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class Watermark:
    """
    在图片右下角添加文字水印的节点。
    支持批量图片，保持与输入相同的Tensor格式（IMAGE）。
    """

    CATEGORY = "NNKK"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_watermark"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "水印文字",
                    },
                ),
            },
            "optional": {
                "font_size": (
                    "INT",
                    {
                        "default": 32,
                        "min": 8,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "margin": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                    },
                ),
                "font_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "可选：字体文件完整路径或字体名，如 C:/Windows/Fonts/msyh.ttc",
                    },
                ),
            },
        }

    def _get_font(self, font_size: int, font_path: str | None = None):
        """
        获取字体：
        - 若用户提供 font_path，优先尝试加载该字体（支持绝对路径或系统字体名）
        - 否则尝试常见系统字体
        - 全部失败时回退到默认字体，保证不会因为字体问题报错
        """
        # 1. 用户指定字体
        if font_path:
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                # 忽略错误，继续走候选字体逻辑
                pass

        # 2. Windows 常见中文字体名可以按需添加
        candidate_fonts = [
            "arial.ttf",  # 通用
            "msyh.ttc",  # 微软雅黑
            "simhei.ttf",  # 黑体
            "simsun.ttc",  # 宋体
        ]

        for font_name in candidate_fonts:
            try:
                return ImageFont.truetype(font_name, font_size)
            except Exception:
                continue

        # 3. 最终兜底：使用Pillow内置字体
        return ImageFont.load_default()

    def _draw_text_bottom_right(self, pil_img: Image.Image, text: str, font, margin: int):
        draw = ImageDraw.Draw(pil_img)

        if not text:
            return pil_img

        # Pillow 新旧版本兼容的文本尺寸获取
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(text, font=font)

        x = max(pil_img.width - text_w - margin, 0)
        y = max(pil_img.height - text_h - margin, 0)

        # 简单的描边效果，提升在浅色背景上的可读性
        outline_range = 1
        for dx in range(-outline_range, outline_range + 1):
            for dy in range(-outline_range, outline_range + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))

        # 主体白色文字
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        return pil_img

    def add_watermark(self, image, text, font_size=32, margin=10, font_path: str = ""):
        """
        核心逻辑：
        - 将ComfyUI的IMAGE Tensor转换为PIL图片
        - 在右下角绘制水印文字
        - 再转换回Tensor返回
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("image 必须为 ComfyUI 的 IMAGE Tensor 类型")

        # 记录设备，便于结果回到原设备
        device = image.device

        # 转到CPU进行处理
        img_np = image.detach().cpu().numpy()

        # 统一为 batch 维度 [B, H, W, C]
        added_batch_dim = False
        if img_np.ndim == 3:
            img_np = img_np[None, ...]
            added_batch_dim = True

        if img_np.ndim != 4:
            raise ValueError(f"不支持的图片维度：{img_np.shape}，期望为 [B, H, W, C]")

        # 处理到 0~255 uint8
        img_np = np.clip(img_np, 0.0, 1.0)
        img_np_uint8 = (img_np * 255.0).round().astype(np.uint8)

        font = self._get_font(int(font_size), font_path.strip() or None)
        margin = int(margin)

        out_list = []
        for i in range(img_np_uint8.shape[0]):
            arr = img_np_uint8[i]
            # 保证是三通道或四通道
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

            pil_img = Image.fromarray(arr)
            pil_img = self._draw_text_bottom_right(pil_img, text, font, margin)

            out_arr = np.array(pil_img).astype(np.float32) / 255.0
            out_list.append(out_arr)

        out_np = np.stack(out_list, axis=0)

        if added_batch_dim:
            out_np = out_np[0]

        out_tensor = torch.from_numpy(out_np).to(device=device)
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "Watermark": Watermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Watermark": "NNKK:Watermark",
}

