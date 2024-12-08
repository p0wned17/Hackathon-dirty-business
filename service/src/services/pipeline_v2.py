import typing as tp

import numpy as np

from src.services.segmentation_detection import MaskGenerator

class SecondPipeline:
    def __init__(
        self,
        segmentation_detector: MaskGenerator,
    ):
        """
        Инициализирует экземпляр класса FirstPipeline, загружая детектор cегментации.

        Args:
            segmentation_detector (MaskGenerator): Модель для детекции cегментации.
        """
        self._segmentation_detector = segmentation_detector

    
    
    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Выполняет предсказание нормализованного cегментации на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            tp.List[str]: Нормализованное изображение cегментации.
        """
        mask = self._segmentation_detector.generate_mask(image)
        
        image_with_mask = self._segmentation_detector.visualize(image, mask)
        
        return image_with_mask