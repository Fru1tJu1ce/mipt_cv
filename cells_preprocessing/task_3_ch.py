import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


def initialize_image(path):
    """Инициализирует картинку"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def mod_rotation(image, angle):
    """Вспомогательная функция вращения для вычисления средней толщины"""
    h, w, _ = image.shape
    center_y, center_x = h // 2, w // 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    cos = np.abs(rotation_matrix[0][0])
    sin = np.abs(rotation_matrix[0][1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0][2] += new_w / 2 - center_x
    rotation_matrix[1][2] += new_h / 2 - center_y
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated


def plot_one_image(image):
    """Выводит картинку"""
    plt.figure(figsize=(7, 7))

    plt.subplot(111)
    plt.title('Сегмент')
    plt.imshow(image)

    plt.show()


def plot_image(image, segmented):
    """Выводит картинку"""
    plt.figure(figsize=(16, 9))

    plt.subplot(121)
    plt.title('Исходное изображение')
    plt.imshow(image)

    plt.subplot(122)
    plt.title('Нарезанное изображение')
    plt.imshow(segmented)

    plt.show()


def calc_mean_width(image):
    """Вычисляем среднюю толщину"""
    width_data = []
    for i in range(0, len(image) - 1, 30):
        width = 0
        for j in range(len(image[0])):
            if (image[i, j] == 255).all() or (image[i, j] == 0).all():
                continue
            else:
                width += 1
        width_data.append(width)

    return np.array(width_data).mean()


def cutting_image(image, value):
    """Нарезает изображение на сегменты"""
    squares = []
    i, j = value // 2, 0
    flag = False
    while i < len(image):
        if flag:
            i += value - 1
            if i >= len(image):
                break
            flag = False
        j = 0
        while j < len(image[0]):
            if (image[i, j] != 255).all():
                flag = True
                square = image[i - value // 2: i + value // 2, j: j + value]
                squares.append(square)
                j += value
            else:
                j += 1
        i += 1

    return squares


def building_new_image(squares, value):
    """Собирает нарезанные кусочки."""

    sqrt = math.ceil(math.sqrt(len(squares)))
    i, j = 0, 0
    ones_array = np.ones((value * sqrt, value * sqrt, 3), dtype=np.uint8)
    ones_array *= 255
    while i < len(ones_array):
        if squares:
            if j < len(ones_array[0]):
                ones_array[i:i + squares[0].shape[0], j:j + squares[0].shape[1]] = squares[0]
                squares.pop(0)
                j += value
            else:
                j = 0
                i += value
        else:
            break

    return ones_array


image = initialize_image('images/train1_1.jpg')
means = []
for i in range(0, 91, 15):
    rotation = mod_rotation(image, i)

    # Сканирование средней ширины
    mean = calc_mean_width(rotation)
    means.append(mean)

min_mean = int(min(means) / 1.5)

# Массив из нарезанных картинок
squares = cutting_image(image, min_mean)

# Собираем кусочки и отображаем
new_img = building_new_image(squares, min_mean)
plot_image(image, new_img)
