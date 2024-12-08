from io import BytesIO

import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File
from fastapi.responses import StreamingResponse

from src.containers.containers import AppContainer
from src.routes.routers import router
from src.services.pipeline_v1 import FirstPipeline
from src.services.utils import decode_jpg, decode_other, encode_jpg


@router.post("/get_predict")
@inject
def get_normalized_image(
    image: bytes = File(),
    service: FirstPipeline = Depends(Provide[AppContainer.segmentation_detector]),
):
    """
    Обрабатывает запрос на получение маски

    Args:
        image (bytes): Входное изображение.
        service (SecondPipeline): Сервис для обработки изображения.
    Returns:
        StreamingResponse: Нормализованное изображение в формате JPEG и предсказанный текст.
    """
    # Проверка на формат JPEG
    if image[:3] == b"\xff\xd8\xff":
        img: np.ndarray = decode_jpg(image)
    else:
        img: np.ndarray = decode_other(image)

    result = service.generate_mask(img)

    # Кодирование изображения в формат JPEG
    jpeg_bytes = encode_jpg(result)

    # Возврат изображения как HTTP-ответ
    return StreamingResponse(
        BytesIO(jpeg_bytes), media_type="image/jpeg"
    )
    