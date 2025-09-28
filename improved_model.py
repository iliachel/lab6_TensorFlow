import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# 1. Prepare Data with Augmentation
DATASET_PATH = 'C:/Users/ichel/.cache/kagglehub/datasets/ritikagiridhar/2000-hand-gestures/versions/3/images'

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
   target_size=(64, 64),
   batch_size=32,
   class_mode='categorical',
   subset='training'
)

# Validation data generator
val_data = val_datagen.flow_from_directory(
   DATASET_PATH,
   target_size=(64, 64),
   batch_size=32,
   class_mode='categorical',
   subset='validation'
)

# 2. Create the Improved CNN Model
model = tf.keras.models.Sequential([
   # Block 1
   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
   tf.keras.layers.MaxPooling2D(2,2),

   # Block 2
   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2,2),
   tf.keras.layers.Dropout(0.25),

   # Block 3
   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2,2),
   tf.keras.layers.Dropout(0.25),

   # Flatten and Dense layers
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(512, activation='relu'),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])


# 3. Compile and Train Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=30) # Increased epochs to 30

# 4. Evaluate Model
loss, accuracy = model.evaluate(val_data)
print(f'Tochnost uluchshennoj modeli: {accuracy * 100:.2f}%')

# 5. Use Model for Prediction
# Find a test image
test_image_path = None
for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
    if filenames:
        test_image_path = os.path.join(dirpath, filenames[0])
        break

if test_image_path:
    print(f"Using test image: {test_image_path}")
    # Load and prepare the image
    img = image.load_img(test_image_path, target_size=(64, 64))

    # Display the image
    plt.imshow(img)
    plt.axis('off') # Hide axes
    plt.title(f"Test Image: {os.path.basename(test_image_path)}")
    plt.show()

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    # Get class name from index
    class_indices = train_data.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class_index]

    print(f'Predskazannyj klass (indeks): {predicted_class_index}')
    print(f'Predskazannyj klass (imya): {predicted_class_name}')
else:
    print("Ne udalos' najti testovoe izobrazhenie.")