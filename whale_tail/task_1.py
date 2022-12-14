import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage


def initialize_picture(path):
    """Инициализирует картинку"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_image(image):
    """Вспомогательная функция для вывода изображения."""
    plt.figure(figsize=(16, 9))

    plt.subplot(111)
    plt.title("Фигура")
    plt.imshow(image, cmap='gray')

    plt.show()


def img_hist_and_plot(image):
    """Строит гистограмму изображения"""
    plt.figure(figsize=(16, 9))

    color = ('r', 'g', 'b')
    data = {}
    plt.subplot(221)
    plt.title("Гистограмма изображения")
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        data[col] = np.argmax(hist)
        plt.plot(hist, color=col)

    for i, col in enumerate(color):
        plt.subplot(2, 2, i + 2)
        plt.title(col + ' спектр')
        plt.imshow(image[:, :, i], cmap='gray')

    plt.show()

    return hist


def plot_images(image, mask):
    """Выводит картинки."""
    plt.figure(figsize=(16, 9))

    plt.subplot(121)
    plt.title("Исходное изображение")
    plt.imshow(image)

    plt.subplot(122)
    plt.title("Маска хвоста")
    plt.imshow(mask, cmap='gray')

    plt.show()


def finding_biggest_cont(contours):
    """Возвращает номер контура с наибольшей площадью"""
    max = 0
    for i, contour in enumerate(contours):
        if max < cv2.contourArea(contour, oriented=False):
            max = cv2.contourArea(contour, oriented=False)
            res = i
    return res


def finding_tail(image):
    """Выделяет маску хвоста на изображении."""
    # Gamma correction
    image = skimage.exposure.adjust_gamma(image, gamma=0.03)

    # Canny + blur + thres
    canny = cv2.Canny(image, 7, 8)
    canny = cv2.GaussianBlur(canny, ksize=(51, 51), sigmaX=2, sigmaY=2)
    canny = cv2.threshold(canny, 1, 255, cv2.THRESH_BINARY_INV)[1]

    # MorphologyEx
    kernel = np.ones((25, 25), np.uint8)
    cor_image = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((10, 10), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_OPEN, kernel)

    # Contours + draw them on the mask
    contours = cv2.findContours(cor_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    num_of_cont = finding_biggest_cont(contours)
    mask = np.zeros(cor_image.shape, dtype=np.uint8)

    # Boxing contour
    rect = cv2.minAreaRect(contours[num_of_cont])
    cv2.drawContours(mask, contours, num_of_cont, (255), -1)

    # draw rect
    img = mask.copy()
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255), 1)
    plot_image(img)

    return mask, rect


def mod_rotation(image, angle):
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


def object_box(object):
    """Помещает объект в коробку"""
    # plot_one_img(object)
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    for i in range(0, len(object) - 1):
        if xmin == 0:
            for j in range(0, len(object[0]) - 1):
                if (object[i, j] != 0).all():
                    xmin = i
                    break
        else:
            break

    for i in range(0, len(object[0]) - 1):
        if ymin == 0:
            for j in range(0, len(object) - 1):
                if (object[j, i] != 0).all():
                    ymin = i
                    break
        else:
            break

    for i in range(len(object) - 1, 0, -1):
        if xmax == 0:
            for j in range(len(object[0]) - 1, 0, -1):
                if (object[i, j] != 0).all():
                    xmax = i
                    break
        else:
            break

    for i in range(len(object[0]) - 1, 0, -1):
        if ymax == 0:
            for j in range(len(object) - 1, 0, -1):
                if (object[j, i] != 0).all():
                    ymax = i
                    break
        else:
            break

    return xmin, ymin, xmax, ymax


image = initialize_picture('images/test_image_10.jpg')
hist = img_hist_and_plot(image)
image_r = image[:, :, 0]

mask, rect = finding_tail(image_r)
angle = rect[2]
print('angle =', angle)
if angle > 45:
    angle = rect[2] - 90
plot_images(image, mask)

masked_image = cv2.bitwise_and(image, image, mask=mask)
masked_image = mod_rotation(masked_image, angle)
xmin, ymin, xmax, ymax = object_box(masked_image)

plot_image(masked_image[xmin: xmax, ymin: ymax])
