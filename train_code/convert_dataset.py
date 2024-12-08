import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

IMAGES_DIR = 'cv_open_dataset/open_img'  # Путь к вашему датасету с изображениями
MASKS_DIR = 'cv_open_dataset/open_msk'  # Путь к вашему датасету с масками
OUTPUT_DIR = 'dataset_for_training_yolo'  # Путь к выходной директории
TRAIN_SIZE = 0.8  # Процент обучающей выборки

os.makedirs(os.path.join(OUTPUT_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/val'), exist_ok=True)

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]
if len(image_files) != len(mask_files):
    print(len(image_files))
    print(len(mask_files))
    print("Количество изображений и масок не совпадает.")
    for mask in mask_files:
        if mask.replace(".png", ".jpg") not in image_files:
            print(mask)
            
train_images, val_images = train_test_split(image_files, train_size=TRAIN_SIZE, random_state=42)

def convert_mask_to_yolo(mask_path, out):
    # Открываем изображение
    image = cv2.imread(mask_path)

    if image is None:
        print(f"Не удалось открыть изображение: {mask_path}")
        return
    
    height, width = image.shape[:2]

    # Создаем маску для черного цвета
    black_mask = cv2.inRange(image, (0, 0, 0), (50, 50, 50))

    # Создаем новое изображение, где черный цвет остается, а остальные цвета становятся белыми
    new_image = np.ones_like(image) * 255  # Начинаем с белого изображения
    new_image[black_mask > 0] = [0, 0, 0]  # Заменяем черные пиксели

    # Преобразуем в градации серого для нахождения контуров
    gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Находим контуры
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сохраняем контуры в текстовом формате
    output_file_name = os.path.splitext(os.path.basename(mask_path))[0] + '.txt'
    output_file_path = os.path.join(OUTPUT_DIR, out, output_file_name)

    with open(output_file_path, 'w') as f:
        for index, contour in enumerate(contours):
            # Получаем координаты всех точек контура
            contour_points = contour.reshape(-1, 2)
            # Нормализуем координаты
            normalized_points = [(x / width, y / height) for x, y in contour_points]
            points_str = ' '.join(f"{x:.3f} {y:.3f}" for x, y in normalized_points)
            f.write(f"0 {points_str}\n")
            
for img in val_images:
    shutil.copy(os.path.join(IMAGES_DIR, img), os.path.join(OUTPUT_DIR, 'images/val', img))
    mask_name = img.replace('.jpg', '.png')
    convert_mask_to_yolo(os.path.join(MASKS_DIR, mask_name), 'labels/val')

