import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Функция загрузки изображений и их меток
def load_images_and_labels(directory):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)

            # Проверка на существование файла
            if os.path.exists(img_path):
                img = cv2.imread(img_path)

                # Проверка на успешное чтение изображения
                if img is not None:
                    img = cv2.resize(img, (128, 128))  # Размер изображения можно изменить по вашему усмотрению
                    images.append(img)

                    # Метка кота (0) или собаки (1) основывается на названии папки
                    label = 0 if "Cat" in directory else 1
                    labels.append(label)

    return np.array(images), np.array(labels)



# Загрузка изображений и меток для тренировочного и тестового набора
train_cat_images, train_cat_labels = load_images_and_labels("Cat")
train_dog_images, train_dog_labels = load_images_and_labels("Dog")

# Объединение изображений и меток
train_images = np.concatenate([train_cat_images, train_dog_images])
train_labels = np.concatenate([train_cat_labels, train_dog_labels])

# Нормализация значений пикселей к диапазону [0, 1]
train_images = train_images / 255.0

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

# Обучение модели
model.fit(train_images, train_labels, epochs=15, batch_size=1)

# Сохранение весов модели
model.save_weights("model_weights.h5")