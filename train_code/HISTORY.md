# Детектор

## TODO

1. Собрать больше данных
2. Уточнить разметку более корректно

______________________________________________________________________

## 08.12.2024: Бейзлайн модель

[Эксперимент](https://app.clear.ml/projects/e880cabaf41141e180b328cde0b99fbb/experiments/644d5112a30c4f15a60f0c85fc8d1052/output/execution)

### Метрики

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│          map50-95(B)      │         0.89              │
└───────────────────────────┴───────────────────────────┘
```

Об эксперименте подробнее можно узнать из аргументов в проекте clearml

## TODO

- Собрать больше данных
- Переписать код, чтобы использовать другие архитектуры
- Подобрать наилучшие гиперпараметры
- Работа с данными 
   
- Улучшение существующей разметки   
- Поиск открытых реальных данных для решения подобной задачи (Soiling Detection dataset https://drive.google.com/drive/folders/1q51oyvx2_SFPts2XgSnKnaVc8ijZ6tKs и другие)  
- Сбор своих реальных данных в "полевых условиях" для решения задачи   
- Скрап данных из интернета
- Поиск классических подходов для генерации синтетики (как raindrop, но другие)   
- Поиск нейросетевых подходов
для генерации синтетики (conditional GAN -like, pix2pix -like, подходы из https://habr.com/ru/companies/nornickel/articles/676296/ или более современные подходы)   
- Что-то другое?   

## 08.12.2024: Бейзлайн модель

[Эксперимент](https://app.clear.ml/projects/e880cabaf41141e180b328cde0b99fbb/experiments/178601621d52474f8f4c0f4cf5a3ef93/output/execution)

### Метрики

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│          map50-95(B)      │         0.6967925         │
└───────────────────────────┴───────────────────────────┘
```

Об эксперименте подробнее можно узнать из аргументов в проекте clearml
