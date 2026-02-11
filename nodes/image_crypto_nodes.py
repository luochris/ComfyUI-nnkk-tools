import hashlib
from typing import Tuple

import numpy as np
import torch


def _image_to_uint8_np(image: torch.Tensor) -> Tuple[np.ndarray, torch.device, bool]:
    """
    将 ComfyUI 的 IMAGE Tensor 统一转换为 uint8 的 numpy 数组。
    返回：(np_uint8[B,H,W,C], 原始device, 是否添加了batch维度)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image 必须为 ComfyUI 的 IMAGE Tensor 类型")

    device = image.device
    img_np = image.detach().cpu().numpy()

    added_batch_dim = False
    if img_np.ndim == 3:
        img_np = img_np[None, ...]
        added_batch_dim = True

    if img_np.ndim != 4:
        raise ValueError(f"不支持的图片维度：{img_np.shape}，期望为 [B, H, W, C]")

    img_np = np.clip(img_np, 0.0, 1.0)
    img_uint8 = (img_np * 255.0).round().astype(np.uint8)
    return img_uint8, device, added_batch_dim


def _uint8_np_to_image(
    img_uint8: np.ndarray, device: torch.device, added_batch_dim: bool
) -> torch.Tensor:
    """
    与 _image_to_uint8_np 相反：把 uint8 数组还原回 IMAGE Tensor。
    """
    out_np = img_uint8.astype(np.float32) / 255.0
    if added_batch_dim and out_np.shape[0] == 1:
        out_np = out_np[0]
    out_tensor = torch.from_numpy(out_np).to(device=device)
    return out_tensor


def _prng_bytes(shape, key: str) -> np.ndarray:
    """
    使用 key 生成稳定的伪随机字节流（numpy）。
    """
    if not key:
        raise ValueError("加解密 key 不能为空")

    # 用 sha256(key) 生成一个稳定的种子
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


class ImageEncrypt:
    """
    将 IMAGE 进行对称加密，输出仍为 IMAGE（可直接保存为 PNG）。
    - 算法：对 uint8 像素做 XOR 流加密，可用相同 key 完全还原。
    - 视觉效果：打开是噪点/灰暗风格的乱图，看不出原图内容。
    """

    CATEGORY = "NNKK"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "encrypt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "加密密钥（字符串，需记住以便解密）",
                    },
                ),
            }
        }

    def encrypt(self, image, key: str):
        img_uint8, device, added_batch_dim = _image_to_uint8_np(image)

        # 生成同形状伪随机字节并 XOR
        rand_bytes = _prng_bytes(img_uint8.shape, key)
        enc_uint8 = np.bitwise_xor(img_uint8, rand_bytes)

        # 为了整体观感更偏灰，可以简单地在可见上稍微压低对比度（非必须）
        # 这里不再做额外变换，保证完全可逆，仅依赖 XOR。

        out_tensor = _uint8_np_to_image(enc_uint8, device, added_batch_dim)
        return (out_tensor,)


class ImageDecrypt:
    """
    将加密后的 IMAGE 使用相同 key 还原成正常图片。
    - 算法：再次 XOR 相同的伪随机流，即可完全恢复原图。
    """

    CATEGORY = "NNKK"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decrypt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "解密密钥（需与加密时一致）",
                    },
                ),
            }
        }

    def decrypt(self, image, key: str):
        # 解密与加密完全相同：再次 XOR 同一伪随机流
        img_uint8, device, added_batch_dim = _image_to_uint8_np(image)
        rand_bytes = _prng_bytes(img_uint8.shape, key)
        dec_uint8 = np.bitwise_xor(img_uint8, rand_bytes)
        out_tensor = _uint8_np_to_image(dec_uint8, device, added_batch_dim)
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "ImageEncrypt": ImageEncrypt,
    "ImageDecrypt": ImageDecrypt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageEncrypt": "NNKK:ImageEncrypt",
    "ImageDecrypt": "NNKK:ImageDecrypt",
}

