import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras import layers, models
import numpy as np
from keras.preprocessing import image
import os

# 1. Prepare Data
DATASET_PATH = 'C:\Users\ichel\.cache\kagglehub\datasets\ritikagiridhar\2000-hand-gestures\versions\3'
IMAGE_SIZE = (96, 96) # MobileNetV2 requires a minimum input size

# ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# The validation generator should not have augmentation
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training data generator
train_data = train_datagen.flow_from_directory(
   DATASET_PATH,
   target_size=IMAGE_SIZE,
   batch_size=32,
   class_mode='categorical',
   subset='training'
)

# Validation data generator
val_data = val_datagen.flow_from_directory(
   DATASET_PATH,
   target_size=IMAGE_SIZE,
   batch_size=32,
   class_mode='categorical',
   subset='validation'
)

# 2. Create the Transfer Learning Model
# Load the MobileNetV2 base model, pre-trained on ImageNet
base_model = MobileNetV2(input_shape=(96, 96, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create the new model on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# 3. Compile and Train the Top Layer
print("--- Training the custom head ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=10)


# 4. Fine-Tune the Model
print("--- Fine-tuning the model ---")
# Unfreeze the base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training
fine_tune_epochs = 10
total_epochs = 10 + fine_tune_epochs

history_fine = model.fit(train_data,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_data)


# 5. Evaluate Model
loss, accuracy = model.evaluate(val_data)
print(f'Точность модели с трансферным обучением: {accuracy * 100:.2f}%')

# 6. Use Model for Prediction
# Find a test image
test_image_path = None
for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
    if filenames:
        test_image_path = os.path.join(dirpath, filenames[0])
        break

if test_image_path:
    print(f"Using test image: {test_image_path}")
    # Load and prepare the image
    img = image.load_img(test_image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    # Get class name from index
    class_indices = train_data.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class_index]

    print(f'Предсказанный класс (индекс): {predicted_class_index}')
    print(f'Предсказанный класс (имя): {predicted_class_name}')
else:
    print("Не удалось найти тестовое изображение.")