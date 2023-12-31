import cv2
import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, BatchNormalization, Concatenate
from keras.utils import to_categorical
from keras.applications import EfficientNetB0
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import random

data_folder = "Animals_detection/"
train_classes_file = data_folder + "train/_classes.txt"
with open(train_classes_file, "r") as f:
    classes = f.read().splitlines()

from Train_get import get_train_data, get_validation_data

if __name__ == '__main__':
    train_images, train_labels = get_train_data()
    validation_images, validation_labels = get_validation_data()


combined_train_data = list(zip(train_images, train_labels))
random.shuffle(combined_train_data)
train_images[:], train_labels[:] = zip(*combined_train_data)

combined_valid_data = list(zip(validation_images, validation_labels))
random.shuffle(combined_valid_data)
validation_images[:], validation_labels[:] = zip(*combined_valid_data)

train_images = np.array(train_images + validation_images)
train_labels = np.array(train_labels + validation_labels)

print('Preprocessing')
output_classes = len(classes)
output_bbox_num = 4
train_labels_categorical = to_categorical(train_labels[:, 0], num_classes=output_classes)
image_width = 224
image_height = 224
train_labels_normalized = train_labels[:, 1:] / np.array([image_width, image_height, image_width, image_height])
train_labels_normalized = np.reshape(train_labels_normalized, (train_labels_normalized.shape[0], 4))
train_labels = np.concatenate((train_labels_normalized, train_labels_categorical), axis=1)

print('Model building')
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)

x1 = Dense(512, activation='relu')(x)
d1 = Dropout(0.2)(x1)
n1 = BatchNormalization()(d1)
x1 = Dense(256, activation='relu')(n1)
d1 = Dropout(0.2)(x1)
n1 = BatchNormalization()(d1)
x1 = Dense(128, activation='relu')(n1)
n1 = BatchNormalization()(x1)
x1 = Dense(64, activation='relu')(n1)
n1 = BatchNormalization()(x1)
class_output = Dense(len(classes), activation='softmax', name='class_output')(n1)

x2 = Dense(512, activation='relu')(x)
d2 = Dropout(0.2)(x2)
n2 = BatchNormalization()(d2)
x2 = Dense(256, activation='relu')(n2)
d2 = Dropout(0.1)(x2)
n2 = BatchNormalization()(d2)
x2 = Dense(128, activation='relu')(n2)
n2 = BatchNormalization()(x2)
x2 = Dense(64, activation='relu')(n2)
n2 = BatchNormalization()(x2)
bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(n2)

print('Create the final model')
combined_output = Concatenate()([bbox_output, class_output])
model = Model(inputs=base_model.input, outputs=combined_output)

print('Загрузка сохраненных весов')
model.load_weights('model_weights')

for layer in base_model.layers:
    layer.trainable = False

print('Compile the model')
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse', 'accuracy'])

print(f'train_images {train_images.shape} train_labels {train_labels.shape}')

print('Train the model')
model.fit(train_images, train_labels, batch_size=32, epochs=5, validation_split=0.1)

model.save_weights('model_weights')

model.save('full_model')