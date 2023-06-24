import cv2
import numpy as np

data_folder = "Animals_detection/"

train_annotations_file = data_folder + "train/_annotations.txt"
train_classes_file = data_folder + "train/_classes.txt"
validation_annotations_file = data_folder + "valid/_annotations.txt"
validation_classes_file = data_folder + "valid/_classes.txt"

print(train_annotations_file)


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


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    return img


def get_train_data():
    train_images = []
    train_labels = []
    _, train_annotations = read_annotations_file(train_annotations_file)
    print('train_annotations')
    for annotation in train_annotations:
        image_path, label, xmin, ymin, xmax, ymax = annotation
        img = preprocess_image(data_folder + '/train/' + image_path)
        train_images.append(img)
        train_labels.append([label, xmin, ymin, xmax, ymax])
    return train_images, train_labels


def get_validation_data():
    validation_images = []
    validation_labels = []
    _, validation_annotations = read_annotations_file(validation_annotations_file)
    print('validation_annotations')
    for annotation in validation_annotations:
        image_path, label, xmin, ymin, xmax, ymax = annotation
        img = preprocess_image(data_folder + '/valid/' + image_path)
        validation_images.append(img)
        validation_labels.append([label, xmin, ymin, xmax, ymax])
    return validation_images, validation_labels


if __name__ == '__main__':
    train_images, train_labels = get_train_data()
    validation_images, validation_labels = get_validation_data()
