import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras import layers, models

# Функция загрузки изображения и его предсказания
def predict_image(file_path):
    # Загрузка предварительно обученных весов
    model.load_weights("model_weights.h5")

    # Загрузка изображения и предобработка
    img = cv2.imread(file_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Предсказание
    prediction = model.predict(img)

    # Вывод результата
    result_label.config(text="Cat" if prediction[0][0] < 0.5 else "Dog")

# Функция для выбора файла через диалоговое окно
def choose_file():
    file_path = filedialog.askopenfilename(title="Choose an image file", filetypes=[("Image files", "*.jpg;*.png")])
    if file_path:
        predict_image(file_path)

# Создание графического интерфейса
root = tk.Tk()
root.title("Cat or Dog Predictor")

# Кнопка для выбора файла
browse_button = tk.Button(root, text="Browse", command=choose_file)
browse_button.pack(pady=20)

# Метка для вывода результата
result_label = tk.Label(root, text="")
result_label.pack(pady=20)

# Создание модели сверточной нейронной сети
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Запуск графического интерфейса
root.mainloop()