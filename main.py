import os
from Train_get import get_train_data, get_validation_data

# Загрузка данных
train_images, train_labels = get_train_data()
validation_images, validation_labels = get_validation_data()

# Остальной код из файла Object_Detection_EfficientNetB0.py

# Создание модели, обучение и сохранение

script_path = 'Object_Detection_EfficientNetB0.py'
n = 5

for _ in range(n):
    os.system('python ' + script_path)
