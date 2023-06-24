import os
import shutil
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import glob

image_folder = "C:\\Users\\User\\PycharmProjects\\Classification_Localization\\training_images"
# xml_folder = "C:\\Users\\User\\PycharmProjects\\Classification_Localization\\training_images"

image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
xml_paths = glob.glob(os.path.join(image_folder, "*.xml"))

image_xml_mapping = {}

# Создаем словарь для сопоставления изображений и соответствующих XML-файлов
for image_path in image_paths:
    image_name = os.path.basename(image_path)
    image_name_without_extension = os.path.splitext(image_name)[0]
    xml_path = os.path.join(image_folder, image_name_without_extension + ".xml")

    if os.path.exists(xml_path):
        image_xml_mapping[image_path] = xml_path

# Разделение на тренировочный и тестовый наборы
all_data = list(image_xml_mapping.items())
random.shuffle(all_data)
num_samples = len(all_data)
num_train_samples = int(0.8 * num_samples)  # 80% для тренировки, 20% для теста

train_data = all_data[:num_train_samples]
test_data = all_data[num_train_samples:]

# Создание папок для тренировочных и тестовых данных
train_folder = "train_data"
test_folder = "test_data"

if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Копирование изображений и соответствующих XML-файлов в соответствующие папки
for image_path, xml_path in train_data:
    shutil.copy(image_path, os.path.join(train_folder, os.path.basename(image_path)))
    shutil.copy(xml_path, os.path.join(train_folder, os.path.basename(xml_path)))

for image_path, xml_path in test_data:
    shutil.copy(image_path, os.path.join(test_folder, os.path.basename(image_path)))
    shutil.copy(xml_path, os.path.join(test_folder, os.path.basename(xml_path)))

# Используйте train_folder и test_folder для дальнейшей обработки тренировочных и тестовых данных
