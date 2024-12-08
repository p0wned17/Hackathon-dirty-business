from ultralytics import YOLO

# Загружаем модель для обучения
model = YOLO("yolov8n-seg.pt")


# Начало обучения
train_results = model.train(
    data="./data.yaml",  # путь к YAML файлу датасета
    epochs=200,  # количество эпох обучения
    imgsz=864,  # увеличиваем размер изображения для лучшей детализации
    device=0,  # устройство для обучения (GPU)
    batch=32,  # увеличиваем размер батча для лучшей стабильности обучения
    pretrained=False,  # использование предобученных весов
    optimizer='SGD',  # оптимизатор
    workers=12,
    cache=True,
    single_cls=True,
    scale=0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0,
    close_mosaic=20,
    mixup=0,
    copy_paste=0,
    erasing=0,
    crop_fraction=0
)

# Сразу можем провалидировать результаты, которые нам запишутся в runs/segment/trainN+1
results = model.val(data="data.yaml", imgsz=640)
print(results)