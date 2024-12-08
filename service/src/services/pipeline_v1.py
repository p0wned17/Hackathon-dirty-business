import typing as tp

import numpy as np

from src.services.segmentation_detection import MaskGenerator

class FirstPipeline:
    def __init__(
        self,
        segmentation_detector: MaskGenerator,
    ):
        """
        Инициализирует экземпляр класса FirstPipeline, загружая детектор ключевых точек и нормализатор номера.

        Args:
            segmentation_detector (MaskGenerator): Модель для детекции ключевых точек.
        """
        self._segmentation_detector = segmentation_detector

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Выполняет предсказание нормализованного номерного знака на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tp.List[str]: Нормализованное изображение номерного знака.
        """
        # Детекция ключевых точек на изображении
        mask = self._keypoints_detector.predict(image)
        
        return mask