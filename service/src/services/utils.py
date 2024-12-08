import cv2
import numpy as np
from fastapi import HTTPException
from turbojpeg import TJPF_RGB, TurboJPEG

# Инициализация TurboJPEG для кодирования и декодирования изображений JPEG
jpeg = TurboJPEG()

def decode_jpg(image_data: bytes) -> np.ndarray:
    """
    Декодирует изображение JPEG из байтового массива.

    Args:
        image_data (bytes): Байтовый массив изображения.

    Returns:
        np.ndarray: Декодированное изображение в формате RGB.

    Raises:
        HTTPException: Если изображение недействительно.
    """
    image = jpeg.decode(image_data, pixel_format=TJPF_RGB)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid JPEG image")
    return image


def decode_other(image_data: bytes) -> np.ndarray:
    """
    Декодирует изображение из байтового массива в формат, отличный от JPEG.

    Args:
        image_data (bytes): Байтовый массив изображения.

    Returns:
        np.ndarray: Декодированное изображение в формате RGB.

    Raises:
        HTTPException: Если изображение недействительно.
    """
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    # Преобразование из BGR в RGB
    return image


def encode_jpg(image_array: np.ndarray) -> bytes:
    """
    Кодирует изображение в формате RGB в JPEG.

    Args:
        image_array (np.ndarray): Массив изображения в формате RGB.

    Returns:
        bytes: Кодированное изображение в формате JPEG.
    """
    return jpeg.encode(image_array)