import cv2
import matplotlib.pyplot as plt
import numpy as np


def initialize_picture(image='lk.jpg'):
    """Преобразует картинку в RGB формат."""
    picture = cv2.imread(image)
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    return picture


def plot_images(image, simple_rotated, mod_rotated):
    """Выводит картинку."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax_1, ax_2, ax_3 = axs

    ax_1.set_title('Исходная картинка', fontsize=15)
    ax_1.imshow(image)
    ax_1.axis('off')

    ax_2.set_title('Аффинные преобразования', fontsize=15)
    ax_2.imshow(simple_rotated)
    ax_2.axis('off')

    ax_3.set_title('Аффинные преобразования modded', fontsize=15)
    ax_3.imshow(mod_rotated)
    ax_3.axis('off')
    plt.show()


def mod_rotation(image, angle):
    h, w, _ = image.shape
    center_y, center_x = h // 2, w // 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    cos = np.abs(rotation_matrix[0][0])
    sin = np.abs(rotation_matrix[0][1])
    print(rotation_matrix)
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0][2] += new_w / 2 - center_x
    rotation_matrix[1][2] += new_h / 2 - center_y
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated

def simple_rotation(image, angle):
    """Поворот без преобразований"""
    h, w, _ = image.shape
    center_y, center_x = h // 2, w // 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated


# Инициализация картинки и ее параметров
image = initialize_picture()
#point = (500, 800)
rotation_angle = 30
print(image.shape)
new_img = mod_rotation(image, rotation_angle)
not_modded = simple_rotation(image, rotation_angle)
plot_images(image, not_modded, new_img)
