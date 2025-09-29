# Импорт необходимых библиотек
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# 1. Подготовка данных с аугментацией
# Указание пути к набору данных
DATASET_PATH = 'C:/Users/ichel/.cache/kagglehub/datasets/ritikagiridhar/2000-hand-gestures/versions/3/images'

# Создание генератора данных с аугментацией для обучающего набора
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Нормализация значений пикселей до диапазона [0, 1]
    validation_split=0.2,      # Резервирование 20% данных для валидации
    rotation_range=40,         # Случайный поворот изображений на угол до 40 градусов
    width_shift_range=0.2,     # Случайный сдвиг изображений по ширине
    height_shift_range=0.2,    # Случайный сдвиг изображений по высоте
    shear_range=0.2,           # Применение сдвиговых преобразований
    zoom_range=0.2,            # Случайное масштабирование изображений
    horizontal_flip=True,      # Случайное отражение изображений по горизонтали
    fill_mode='nearest'        # Заполнение пикселей за пределами границ изображения
)

# Создание генератора данных для валидационного набора (без аугментации)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


# Создание генератора обучающих данных
train_data = train_datagen.flow_from_directory(
   DATASET_PATH,               # Путь к набору данных
   target_size=(64, 64),       # Изменение размера всех изображений до 64x64 пикселей
   batch_size=32,              # Размер пакета данных
   class_mode='categorical',   # Режим классификации (категориальный)
   subset='training'           # Указание, что это обучающий набор
)

# Создание генератора валидационных данных
val_data = val_datagen.flow_from_directory(
   DATASET_PATH,               # Путь к набору данных
   target_size=(64, 64),       # Изменение размера всех изображений до 64x64 пикселей
   batch_size=32,              # Размер пакета данных
   class_mode='categorical',   # Режим классификации (категориальный)
   subset='validation'         # Указание, что это валидационный набор
)

# 2. Создание улучшенной модели сверточной нейронной сети (CNN)
model = tf.keras.models.Sequential([
   # Блок 1
   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)), # Сверточный слой с 32 фильтрами
   tf.keras.layers.MaxPooling2D(2,2),                                            # Слой подвыборки (пулинга)

   # Блок 2
   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                         # Сверточный слой с 64 фильтрами
   tf.keras.layers.MaxPooling2D(2,2),                                            # Слой подвыборки
   tf.keras.layers.Dropout(0.25),                                                # Слой Dropout для регуляризации

   # Блок 3
   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),                        # Сверточный слой с 128 фильтрами
   tf.keras.layers.MaxPooling2D(2,2),                                            # Слой подвыборки
   tf.keras.layers.Dropout(0.25),                                                # Слой Dropout для регуляризации

   # Полносвязные слои
   tf.keras.layers.Flatten(),                                                    # Преобразование данных в одномерный массив
   tf.keras.layers.Dense(512, activation='relu'),                                # Полносвязный слой с 512 нейронами
   tf.keras.layers.Dropout(0.5),                                                 # Слой Dropout для регуляризации
   tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')   # Выходной слой с функцией активации softmax
])


# 3. Компиляция и обучение модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Компиляция модели
model.fit(train_data, validation_data=val_data, epochs=30)                          # Обучение модели на 30 эпохах

# 4. Оценка модели
loss, accuracy = model.evaluate(val_data) # Оценка модели на валидационных данных
print(f'Tochnost uluchshennoj modeli: {accuracy * 100:.2f}%')

# 5. Использование модели для предсказания
# Поиск тестового изображения
test_image_path = None
for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
    if filenames:
        test_image_path = os.path.join(dirpath, filenames[0])
        break

if test_image_path:
    print(f"Using test image: {test_image_path}")
    # Загрузка и подготовка изображения
    img = image.load_img(test_image_path, target_size=(64, 64))

    # Отображение изображения
    plt.imshow(img)
    plt.axis('off') # Скрытие осей
    plt.title(f"Test Image: {os.path.basename(test_image_path)}")
    plt.show()

    # Преобразование изображения в массив и нормализация
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Добавление измерения для пакета

    # Сделать предсказание
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction) # Получение индекса предсказанного класса

    # Получение имени класса по индексу
    class_indices = train_data.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class_index]

    print(f'Predskazannyj klass (indeks): {predicted_class_index}')
    print(f'Predskazannyj klass (imya): {predicted_class_name}')
else:
    print("Ne udalos' najti testovoe izobrazhenie.")