import base64
import io
from PIL import Image
from typing import Union

class ImageWrapper:
    """
    统一包装图像
    支持按需在 Base64 字符串和 PIL.Image 之间进行转换，且自带缓存避免重复计算
    """
    def __init__(self, image_data: Union[str, Image.Image]):
        self._pil_image = None
        self._base64_str = None

        if isinstance(image_data, Image.Image):
            self._pil_image = image_data
        elif isinstance(image_data, str):
            if image_data.startswith("data:image"):
                self._base64_str = image_data.split(",", 1)[1]
            else:
                self._base64_str = image_data
        else:
            raise ValueError("Unsupported image data type. Must be PIL.Image or Base64 string.")
        
    def as_pil(self) -> Image.Image:
        if self._pil_image is None:
            image_bytes = base64.b64decode(self._base64_str)
            self._pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._pil_image
    
    def as_base64(self) -> str:
        if self._base64_str is None:
            buffered = io.BytesIO()
            self._pil_image.save(buffered, format="PNG")
            self._base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return self._base64_str
    
    def as_bytes(self) -> bytes:
        if self._pil_image is not None:
            buffered = io.BytesIO()
            self._pil_image.save(buffered, format="PNG")
            return buffered.getvalue()
        elif self._base64_str is not None:
            return base64.b64decode(self._base64_str)
        else:
            raise ValueError("No valid image data available.")
    
    def as_data_url(self, mime_type: str = "image/png") -> str:
        b64_str = self.as_base64()
        return f"data:{mime_type};base64,{b64_str}"
