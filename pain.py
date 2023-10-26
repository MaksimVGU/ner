import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tkinter import *
from PIL import Image, ImageDraw

# Загрузка и предобработка данных
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Создание модели
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Загрузка и сохранение весов
WEIGHTS_FILE = 'mnist_weights.h5'
try:
    model.load_weights(WEIGHTS_FILE)
    print("Загружены веса модели из", WEIGHTS_FILE)
except:
    print("Обучение модели...")
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    model.save_weights(WEIGHTS_FILE)
    print(f"Веса модели сохранены в {WEIGHTS_FILE}")

def test_drawn_image():
    image = Image.new("L", (560, 560), (255))
    draw = ImageDraw.Draw(image)

    def paint(event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)  # изменено на черный цвет
        draw.line([x1, y1, x2, y2], fill="black", width=5)

    def test_model():
        img_resized = image.resize((28, 28))
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        img_array = np.abs(1 - img_array)
        img_array = img_array.reshape((1, 28, 28, 1))
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        label.config(text=f"Predicted: {predicted_class}")

    def clear_canvas():
        canvas.delete("all")
        draw.rectangle([0, 0, 560, 560], fill=(255))

    def train_on_image():
        correct_label = int(entry.get())
        img_resized = image.resize((28, 28))
        img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
        img_array = np.abs(1 - img_array)
        img_array = img_array.reshape((1, 28, 28, 1))
        correct_label_array = tf.keras.utils.to_categorical(correct_label, num_classes=10)
        model.fit(img_array, np.array([correct_label_array]), epochs=1)
        label.config(text=f"Обучено на: {correct_label}")

    root = Tk()
    root.title("Draw a Number")

    canvas = Canvas(root, bg="white", width=560, height=560)
    canvas.pack(pady=20)
    canvas.bind("<B1-Motion>", paint)

    test_button = Button(root, text="Test", command=test_model)
    test_button.pack(pady=20)

    clear_button = Button(root, text="Clear", command=clear_canvas)
    clear_button.pack(pady=20)

    entry = Entry(root, width=5)
    entry.pack(pady=20)
    entry.insert(0, "0")

    train_button = Button(root, text="Обучить", command=train_on_image)
    train_button.pack(pady=20)

    label = Label(root, text="Predicted:")
    label.pack(pady=20)

    root.mainloop()

test_drawn_image()
    