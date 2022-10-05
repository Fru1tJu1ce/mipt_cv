import cv2
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.set_printoptions(threshold=np.inf)


def cross_correlation(image, template):
    """Fast convolution into correlation"""
    Hk, Wk = template.shape
    Hi, Wi = image.shape
    out = np.zeros((Hi, Wi))

    for i in range(Hi - Hk + 1):
        for j in range(Wi - Wk + 1):
            out[i + Hk // 2, j + Wk // 2] = np.sum(image[i: i + Hk, j: j + Wk] * template)
    return out


def match_template(image_gray, template):
    """Встроенная функция корреляции в cv2"""
    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    plt.figure(figsize=(15, 5))
    plt.imshow(res, cmap='gray')
    plt.title('Карта корреляции')
    plt.show()


def zero_mean_cross_correlation(image, template):
    """Нормировка по среднему значению шаблона"""
    Hk, Wk = template.shape
    Hi, Wi = image.shape
    out = np.zeros((Hi, Wi))
    mean = np.mean(template)
    template = template - mean

    for i in range(Hi - Hk + 1):
        for j in range(Wi - Wk + 1):
            out[i + Hk // 2, j + Wk // 2] = np.sum(image[i: i + Hk, j: j + Wk] * template)
    return out


def draw_correlation(temp, image, cc_matrix, x, y, name):
    """Выводит графики корреляции"""
    plt.figure(figsize=(20, 15))
    plt.subplot(311), plt.imshow(temp), plt.title('Template'), plt.axis('off')

    plt.subplot(312), plt.imshow(image), plt.title('Image'), plt.axis('off')
    plt.plot(x, y, 'rx', ms=40, mew=10)

    plt.subplot(313), plt.imshow(cc_matrix), plt.title(name), plt.axis('off')


    plt.show()


def normalized_cross_correlation(image, template):
    """Нормированная кросс-корреляция"""
    Hk, Wk = template.shape
    Hi, Wi = image.shape
    out = np.zeros((Hi, Wi))
    mean = np.mean(template)
    template = (template - mean) / np.std(template)

    for i in range(Hi - Hk + 1):
        for j in range(Wi - Wk + 1):
            a = image[i: i + Hk, j: j + Wk]
            out[i + Hk // 2, j + Wk // 2] = np.sum((((a - np.mean(a)) / np.std(a)) * template))
    return out


image = cv2.imread('images/shelf.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_grey = cv2.imread('images/shelf.jpg', 0)
temp = cv2.imread('images/template.jpg')
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
temp_grey = cv2.imread('images/template.jpg', 0)


cross_correlation_matrix = cross_correlation(image_grey, temp_grey)
y, x = np.unravel_index(cross_correlation_matrix.argmax(), cross_correlation_matrix.shape)
draw_correlation(temp, image, cross_correlation_matrix, x, y, 'Cross-correlation')

cross_correlation_matrix = zero_mean_cross_correlation(image_grey, temp_grey)
y, x = np.unravel_index(cross_correlation_matrix.argmax(), cross_correlation_matrix.shape)
draw_correlation(temp, image, cross_correlation_matrix, x, y, 'Zero mean cross-correlation')


image = cv2.imread('images/shelf_dark.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_grey = cv2.imread('images/shelf_dark.jpg', 0)

out = normalized_cross_correlation(image_grey, temp_grey)
y, x = np.unravel_index(out.argmax(), out.shape)
draw_correlation(temp, image, out, x, y, 'Normalized cross-correlation')
