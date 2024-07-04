import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, models, optimizers, applications, preprocessing
import numpy as np
import pandas as pd
import os
import time
import json

import matplotlib.pyplot as plt

# Hyperparameters
num_epochs = 100
batch_size = 32
learning_rate = 0.0001
momentum = 0.9
img_size = 224

# Define the transformation pipeline
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

train_datagen = preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    rescale=1./255,
    zoom_range=0.2,
    brightness_range=[0.9, 1.1],
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=lambda x: (x - mean) / std
)

val_test_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: (x - mean) / std
)

# Import Data
data_dir = 'Soybean_ML_orig'
# data_dir = '/Users/oalabi1/Desktop/PhD/Datasets/Soybean_ML_orig_20'
sets = ['train', 'val', 'test']

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

# Setup pretrained model with ImageNet's pretrained weights
base_model = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# print(base_model.summary())

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# model summary
model.summary()

# Compile the model
model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # Compile the model
# model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/best_model_tf.keras",
        verbose=1,
        save_best_only=True,
        monitor="val_loss"
    )
]


# Train the model
training_start_time = time.time()  # Start time for the epoch

history_1 = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    callbacks=callbacks
)


training_end_time = time.time()  # End time for the epoch
training_duration = training_end_time - training_start_time  # Duration of the epoch

print("Total trining duration", training_duration, "seconds.")


# Saving the class names as json file

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history_1.history) 

# save to json:  
hist_json_file = 'history_tf.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


# Save best model to Google Drive
model.save('saved_models/final_model_tf.keras')

# Evaluate the model
model.evaluate(test_generator)


# # Plot training & validation accuracy values
# plt.plot(history_1.history['accuracy'])
# plt.plot(history_1.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Number of Epochs')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history_1.history['loss'])
# plt.plot(history_1.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Number of Epochs')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()