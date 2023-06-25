import os
from Train_get import get_train_data, get_validation_data

<<<<<<< HEAD
# Загрузка данных
train_images, train_labels = get_train_data()
validation_images, validation_labels = get_validation_data()


script_path = 'Object_Detection_EfficientNetB0.py'
n = 8
=======
if __name__ == '__main__':
    train_images, train_labels = get_train_data()
    validation_images, validation_labels = get_validation_data()
    print(train_images.shape)
    print(train_labels.shape)
    print(validation_images.shape)
    print(validation_labels.shape)

script_path = 'Object_Detection_EfficientNetB0.py'
n = 5
>>>>>>> origin/master

for _ in range(n):
    os.system('python ' + script_path)
