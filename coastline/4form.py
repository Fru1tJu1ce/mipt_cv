#from skimage.color import label2rgb
import matplotlib.pyplot as plt
import skimage.io

import sklearn
from sklearn.cluster import AgglomerativeClustering

import numpy as np
import os
import cv2

import warnings

warnings.simplefilter('ignore')


def apply_filters(image: np.ndarray, n_gauss: int = 1, gamma: float = None) -> np.ndarray:
    """
    Применяет GaussianBlur n_gauss раз.
    Опционально применяет gamma-filtering
    """
    for i in range(n_gauss):
        image = cv2.GaussianBlur(image, ksize=(51, 51), sigmaX=2, sigmaY=2)
    if gamma:
        image = skimage.exposure.adjust_gamma(image, gamma=gamma)

    return image


def initialize_picture(path: str):
    """Инициализирует картинку"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def plot_images(image: np.ndarray, img: np.ndarray, n_clusters, label, cmap: str = None) -> None:
    """Отображает картинку"""
    plt.figure(figsize=(16, 9))

    plt.subplot(121)
    plt.title("Картинка")
    plt.imshow(image)
    c_map = plt.cm.get_cmap('Spectral', n_clusters)
    rgba = c_map(np.linspace(0, 1, n_clusters))
    for l in range(n_clusters):
        tmp = label == l
        plt.contour(tmp[:, :], contours=1,
                    colors=[rgba[l]])
    plt.xticks(())
    plt.yticks(())

    plt.subplot(122)
    plt.title("Маска")
    plt.imshow(img, cmap=cmap)
    plt.xticks(())
    plt.yticks(())

    plt.show()


def plot_image(image: np.ndarray, cmap: str = None) -> None:
    """Отображает картинку"""
    plt.figure(figsize=(16, 9))

    plt.title("Картинка")
    plt.imshow(image, cmap=cmap)

    plt.show()


def list_of_imgs(imgs_dir: str) -> list:
    """Возвращает список картинок из директории"""
    images = list()
    # Перебираем картинки из директории с картинками
    for i, img in enumerate(os.listdir(imgs_dir)):
        path = os.path.join(imgs_dir, img)
        image = initialize_picture(path)
        images.append(image)
    return images


images = list_of_imgs('images')
for img in images:
    # Blurring & gamma
    n_img = img.copy()
    n_img = apply_filters(n_img, n_gauss=15)

    X = np.reshape(n_img, (-1, 1))
    image_ = skimage.color.rgb2hsv(n_img)
    X = np.reshape(image_[:, :, 0], (-1, 1))

    n_clusters = 2  # number of regions
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++',
                                    n_init=40, algorithm='elkan',
                                    max_iter=1000).fit(X)
    label = np.reshape(kmeans.labels_, img[:, :, 0].shape)

    kernel = np.ones((2, 2), np.uint8)
    label = label.astype(np.uint8)
    label = cv2.morphologyEx(label, cv2.MORPH_CLOSE, kernel)
    label = cv2.morphologyEx(label, cv2.MORPH_OPEN, kernel)

    # Вода - темная
    # Берег - белый
    # Наложить маску и посмотреть на соотношение цветов. У воды опытным путем установлено, что 0.4 <red/green < 0.7
    # -> решение: (инвертировать маску или нет?)

    bitwise = cv2.bitwise_and(img, img, mask=label)
    mask_mean = cv2.mean(img, label)
    print(mask_mean)
    if mask_mean[0] / mask_mean[1] > 0.4 and mask_mean[0] / mask_mean[1] < 0.7:
        label = ~label

    # Plot the results on an image
    plot_images(img, label, n_clusters, label, cmap='gray')
