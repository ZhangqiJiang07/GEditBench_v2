from __future__ import annotations

import os
import base64
from io import BytesIO
from typing import Union

from PIL import Image

def check_image_exists(image_path: str) -> bool:
    """ Check if an image exists at the given path, supporting both local and S3 paths. """
    if image_path.startswith('s3://'):
        import megfile
        return megfile.smart_glob(image_path) is not None
    return os.path.exists(image_path)

def open_image(image_path: Union[str, Image.Image, bytes]) -> Image.Image:
    if isinstance(image_path, Image.Image):
        return image_path.convert("RGB") if image_path.mode != "RGB" else image_path
    if isinstance(image_path, bytes):
        return Image.open(BytesIO(image_path)).convert("RGB")
    if isinstance(image_path, str):
        if image_path.startswith("s3://"):
            import megfile

            with megfile.smart_open(image_path, "rb") as f:
                return Image.open(f).convert("RGB")
        return Image.open(image_path).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image_path)}")

def compress_convert_image2any(
    image_pil: Image.Image,
    max_side: int = None,
    target_type: str = None,
):
    assert isinstance(image_pil, Image.Image), "input image should be PIL Image"
    image = image_pil.copy().convert("RGB")
    if max_side is not None:
        image.thumbnail((max_side, max_side))

    if target_type is None:
        return image
    if target_type == "bytes":
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    if target_type == "url":
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    raise ValueError("Only support target_type: None, bytes, url")

def image_to_data_url(image_pil, max_side: int = 1024) -> str:
    if isinstance(image_pil, str):
        image_pil = open_image(image_pil)
    image = image_pil.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def image_to_data_url_no_resize(image_pil) -> str:
    if isinstance(image_pil, str):
        image_pil = open_image(image_pil)
    image = image_pil.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
