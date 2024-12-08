import cv2
import numpy as np
from ultralytics import YOLO 
import typing as tp


class MaskGenerator:
    def __init__(self, config: tp.Dict):
        # Инициализация модели YOLO
        self.model = YOLO(config['model_path'])

    def infer_image(self, image):
        """
        Выполняет инференс на входном изображении.
        """
        # Выполняем инференс
        return self.model(image, verbose=False)

    def create_mask(self, image, results):
        """
        Создаёт сглаженную маску на основе результатов инференса.
        """
        height, width = image.shape[:2]

        # Создаем пустую маску с черным фоном
        mask = np.zeros((height, width), dtype=np.uint8)

        # Проходим по результатам и создаем маску
        for result in results:
            masks = result.masks  # Получаем маски из результатов
            if masks is not None:
                for mask_array in masks.data:  # Получаем маски как массивы
                    mask_i = mask_array.cpu().numpy()  # Преобразуем маску в numpy массив

                    # Изменяем размер маски под размер оригинального изображения
                    mask_i_resized = cv2.resize(
                        mask_i, (width, height), interpolation=cv2.INTER_LINEAR
                    )

                    # Накладываем обработанную маску на финальную маску
                    mask[mask_i_resized > 0] = 255

        return mask
    
    def visualize(self, source_image, mask):
        """
        Параметры:
            - source_image (numpy.array): Оригинальная картинка
            - mask(numpy.array): Сгенерированная маска
        Возвращает:
            - overlay (numpy.array): Наложенная маска на картинку
        """
        overlay = cv2.addWeighted(source_image, 0.7, mask, 0.3, 0)

        return overlay
    
    def generate_mask(self, image):
        """
        Выполняет полный процесс: инференс и создание маски.
        """
        # Выполняем инференс
        results = self.infer_image(image)
        # Создаем маску
        mask = self.create_mask(image, results)
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        return mask