# Face & Hand Detection with Identity and Emotion Recognition

Этот проект использует `OpenCV`, `MediaPipe` и `DeepFace` для:
- определения лица в кадре с веб-камеры,
- верификации пользователя по фотографии,
- подсчёта пальцев на руке,
- отображения имени, фамилии и текущей эмоции.

## Функциональность

- ✅ Распознавание лица в реальном времени
- ✅ Верификация пользователя с помощью DeepFace
- ✅ Подсчёт количества поднятых пальцев
- ✅ Реакция на число пальцев:
  - **1 палец** → имя
  - **2 пальца** → фамилия
  - **3 пальца** → имя + фамилия + эмоция
- ✅ Отображение информации на экране


## Демонстрация

![example1](https://github.com/user-attachments/assets/7f1a7f79-9ee9-4441-ae82-0f8921adb892)
![example2](https://github.com/user-attachments/assets/d1dc7d91-02e8-4509-a7e3-4c4d0b4c8e20)
![example3](https://github.com/user-attachments/assets/b872696f-3953-405b-b7db-96217b92a4ae)

      
## Используемые технологии

- Python 3.8
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [DeepFace](https://github.com/serengil/deepface)
