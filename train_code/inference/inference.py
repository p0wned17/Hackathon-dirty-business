from ultralytics import YOLO
import cv2
import numpy as np

# Здесь лучше указывать свою получившиюся модель для определения маски
model = YOLO("some_path_to_best_model")


def infer_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Инференс
    return model(image)


def create_smooth_mask(image_path, results):
    # Загружаем изображение и получаем его размеры
    image = cv2.imread(image_path)
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


image = "path_to_some_image"

results = infer_image(image)
mask_image = create_smooth_mask(image, results)

# Сохраняем маску в формате PNG
mask_output_path = "./mask_image.png"  # Укажите путь для сохранения маски
cv2.imwrite(mask_output_path, mask_image)
