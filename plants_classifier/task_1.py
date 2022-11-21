import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from sklearn.linear_model import LogisticRegression

import warnings

warnings.simplefilter('ignore')


def initialize_picture(path):
    """Инициализирует изображение."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image


def train_labels(train_dir):
    filename = 'labels.csv'
    with open(filename, 'w', newline="") as f_o:
        rowwriter = csv.writer(f_o)
        for num, category in enumerate(os.listdir(train_dir)):
            path = os.path.join(train_dir, category)
            for img in os.listdir(path):
                rowwriter.writerow([img, num])


def plot_image(image, name='Исходная картинка'):
    """Выводит картинку"""
    plt.figure(figsize=(16, 9))

    plt.subplot(111)
    plt.title(name)
    plt.imshow(image, cmap='gray')

    plt.show()


def find_green(image):
    """Ищет зеленые области на изображении"""
    image_mask = cv2.inRange(image, (29, 40, 40), (75, 255, 135))
    kernel = np.ones((4, 4))
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3))
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)
    return image_mask


def find_features(mask):
    """Возвращает фичи"""
    P = 0
    S = 0

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Ищем контуры, сохраняем периметр и площадь.
    for contour in contours:
        if cv2.contourArea(contour, oriented=False) > 500:
            P += cv2.arcLength(contour, True)
            S += cv2.contourArea(contour, oriented=False)

    return P, S


def generating_x(X, dir_x):
    """Возвращает датафрейм с фичами"""
    for i, img_name in enumerate(os.listdir(dir_x)):
        # Контуры + площадь с периметром
        image = initialize_picture(dir_x + '/' + img_name)
        mask = find_green(image)
        P, S = find_features(mask)

        # Записываем результат в датафрейм X
        df_dict = pd.DataFrame({'perimeter': P, 'area': S}, index=[i])
        X = pd.concat([X, df_dict])

    return X


# creating csv file with labels
# Data dir
datadir = 'images'
# Dir with classified folders(у папок есть имена, они являются классами)
train_dir = datadir + '/train/classified'
train_labels(train_dir)

# image dirs
datadir = 'images'
train_dir = datadir + '/train/all_images'
test_dir = datadir + '/test'

X_train = pd.DataFrame()
Y_train = pd.DataFrame()
X_test = pd.DataFrame()

# Opening csv file with target data and writing data to Y_train
with open('labels.csv', 'r') as f_o:
    reader = csv.reader(f_o)
    for i, row in enumerate(reader):
        df_dict = pd.DataFrame({'class': row[1]}, index=[i])
        Y_train = pd.concat([Y_train, df_dict])
# Labels of classes
dict_class = {'0': 'Loose Silky-bent', '1': 'Maize', '2': 'Scentless Mayweed', '3': 'Small-flowered Cranesbill'}

# Data pre-process
X_train = generating_x(X_train, train_dir)
X_test = generating_x(X_test, test_dir)

# Fitting model
lr = LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=-1)
lr.fit(X_train, Y_train)
print('score:', lr.score(X_train, Y_train))

# Predicting
Y_pred = lr.predict(X_test)

# Plotting images with predicted names (36/40 are correct)
for i, img_name in enumerate(os.listdir(test_dir)):
    image = cv2.imread(test_dir + '/' + img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plot_image(image, dict_class[Y_pred[i]])
