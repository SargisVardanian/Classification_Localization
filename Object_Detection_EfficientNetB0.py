from keras.models import load_model
from keras.preprocessing import image
import glob
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.applications import MobileNetV2
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

train_annotations_file = data_folder + "train/_annotations.txt"
train_classes_file = data_folder + "train/_classes.txt"
test_annotations_file = data_folder + "test/_annotations.txt"
test_classes_file = data_folder + "test/_classes.txt"
validation_annotations_file = data_folder + "valid/_annotations.txt"
validation_classes_file = data_folder + "valid/_classes.txt"

print(train_annotations_file)

with open(train_classes_file, "r") as f:
    classes = f.read().splitlines()


def read_annotations_file(annotations_file):
    annotations = []
    with open(annotations_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            values = line.split(" ")
            image_path = values[0]
            bboxes = values[1:]
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, label = map(int, bbox.split(","))
                annotations.append([image_path, label, xmin, ymin, xmax, ymax])
    return image_path, annotations



train_path, train_annotations = read_annotations_file(train_annotations_file)
test_path, test_annotations = read_annotations_file(test_annotations_file)
valid_path, validation_annotations = read_annotations_file(validation_annotations_file)



train_images = []
train_labels = []
print('train_annotations')
for annotation in train_annotations:
    # p = data_folder + '/train/' + train_path
    # print('p', p)
    image_path, label, xmin, ymin, xmax, ymax = annotation
    img = cv2.imread(data_folder + '/train/' + image_path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    train_images.append(img)
    train_labels.append([label, xmin, ymin, xmax, ymax])

test_images = []
test_labels = []
print('test_annotations')
for annotation in test_annotations:
    image_path, label, xmin, ymin, xmax, ymax = annotation
    img = cv2.imread(data_folder + '/test/' + test_path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    test_images.append(img)
    test_labels.append([label, xmin, ymin, xmax, ymax])

validation_images = []
validation_labels = []
print('validation_annotations')
for annotation in validation_annotations:
    image_path, label, xmin, ymin, xmax, ymax = annotation
    img = cv2.imread(data_folder + '/valid/' + valid_path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    validation_images.append(img)
    validation_labels.append([label, xmin, ymin, xmax, ymax])

combined_train_data = list(zip(train_images, train_labels))
random.shuffle(combined_train_data)
train_images[:], train_labels[:] = zip(*combined_train_data)

combined_valid_data = list(zip(validation_images, validation_labels))
random.shuffle(combined_valid_data)
validation_images[:], validation_labels[:] = zip(*combined_valid_data)

train_images = np.array(train_images + validation_images)
train_labels = np.array(train_labels + validation_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print('Preprocessing')
output_classes = len(classes)
output_bbox_num = 4

train_labels_categorical = to_categorical(train_labels[:, 0], num_classes=output_classes)

image_width = 224
image_height = 224

print('Normalize bounding box coordinates')
train_labels_normalized = train_labels[:, 1:] / np.array([image_width, image_height, image_width, image_height])
print('train_labels_normalized', train_labels_normalized.shape)

print('Reshape bounding box coordinates to (num_samples, 4)')
train_labels_normalized = np.reshape(train_labels_normalized, (train_labels_normalized.shape[0], 4))
print('train_labels_normalized', train_labels_normalized.shape)

print('Concatenate categorical labels and normalized bounding box coordinates')
train_labels = np.concatenate((train_labels_normalized, train_labels_categorical), axis=1)
print('train_labels', train_labels.shape, train_images.shape)

print('Model building')
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
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
d2 = Dropout(0.2)(x2)
n2 = BatchNormalization()(d2)
x2 = Dense(128, activation='relu')(n2)
n2 = BatchNormalization()(x2)
x2 = Dense(64, activation='relu')(n2)
n2 = BatchNormalization()(x2)
bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(n2)

print('Create the final model')
combined_output = Concatenate()([bbox_output, class_output])
print('output', combined_output)
model = Model(inputs=base_model.input, outputs=combined_output)

# print('Загрузка сохраненных весов')
# model.load_weights('model_weights')

# Установка слоев как необучаемые
for layer in base_model.layers:
    layer.trainable = False

print('Compile the model')
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse', 'accuracy'])

print(f'train_images {train_images.shape} train_labels {train_labels.shape}')

print('Train the model')
model.fit(train_images, train_labels, batch_size=64, epochs=5, validation_split=0.1)

model.save_weights('model_weights')

model.save('full_model')
