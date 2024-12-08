from ultralytics import YOLO

# Загружаем модель для обучения
model = YOLO("yolov8n-seg.pt")


# Начало обучения
train_results = model.train(
    data="./data.yaml",  # путь к YAML файлу датасета
    epochs=100,  # количество эпох обучения
    imgsz=640,  # увеличиваем размер изображения для лучшей детализации
    device=0,  # устройство для обучения (GPU)
    batch=16,  # увеличиваем размер батча для лучшей стабильности обучения
    pretrained=False,  # использование предобученных весов
    optimizer='SGD',  # оптимизатор
    workers=12, # количество воркеров
    single_cls=True, # количество классов, в данном случае у нас он единственный, поэтому ставим single_cls=True
)

# Сразу можем провалидировать результаты, которые нам запишутся в runs/segment/trainN+1
results = model.val(data="./data.yaml", imgsz=640)
print(results)