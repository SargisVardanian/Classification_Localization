import random
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

# Загрузка модели с весами
# model = load_model('full_model')
model = tf.saved_model.load('full_model')

data_folder = "Animals_detection/"

test_annotations_file = data_folder + "test/_annotations.txt"
test_classes_file = data_folder + "test/_classes.txt"

with open(test_classes_file, "r") as f:
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


test_path, test_annotations = read_annotations_file(test_annotations_file)

test_images = []
test_labels = []
print('test_annotations')
for annotation in test_annotations:
    image_path, label, xmin, ymin, xmax, ymax = annotation
    img = cv2.imread(data_folder + '/test/' + image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Исправленная строка
    img = cv2.resize(img, (224, 224))
    test_images.append(img)
    test_labels.append([label, xmin, ymin, xmax, ymax])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

test_images_tensor = tf.convert_to_tensor(test_images, dtype=tf.float32)

output = model(test_images_tensor)
n = 10
class_predictions = output[:, 4:]
bbox_predictions = output[:, :4]

predicted_class = np.argmax(class_predictions[n])
predicted_bbox = bbox_predictions[n]
test_labels = test_labels[n]

print("Predicted Class:", predicted_class)
print("Predicted Bounding Box:", predicted_bbox)
print('test_images', test_images.shape)
print('test_labels', test_labels)
_, image_height, image_width, _ = test_images.shape

x_min = int(predicted_bbox[0] * image_width)
y_min = int(predicted_bbox[1] * image_height)
x_max = int(predicted_bbox[2] * image_width)
y_max = int(predicted_bbox[3] * image_height)

old_x_min = test_labels[1]
old_y_min = test_labels[2]
old_x_max = test_labels[3]
old_y_max = test_labels[4]

image_copy = np.copy(test_images[n])
class_label = classes[predicted_class]
actual_class_label = classes[test_labels[0]]

cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.rectangle(image_copy, (old_x_min, old_y_min), (old_x_max, old_y_max), (255, 0, 0), 2)

plt.title('Predicted Class: {} | Actual Class: {}'.format(class_label, actual_class_label))
plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
plt.show()
