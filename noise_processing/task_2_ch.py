import matplotlib.pyplot as plt
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)


def initialize_image(path):
    """Инициализирует картинку для дальнейшей обработки"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def plot_one_img(object):
    """Выводит одну картинку"""
    plt.figure(figsize=(16, 9))
    plt.subplot(111)
    plt.title("Бинаризованное изображение")
    plt.imshow(object, cmap='gray')
    plt.show()


def plot_images(image, triangle, circle, background, target):
    """Отрисовывает картинку"""
    plt.figure(figsize=(16, 9))

    # Выводим основное изображение
    plt.subplot(321)
    plt.title("Исходное изображение")
    plt.imshow(image, cmap='gray')

    # Выводим треугольник
    plt.subplot(322)
    plt.title("Треугольник на изображении")
    plt.imshow(triangle, cmap='gray')

    # Выводим круг
    plt.subplot(323)
    plt.title("Круг на изображении")
    plt.imshow(circle, cmap='gray')

    # Выводим фон
    plt.subplot(324)
    plt.title("Фон на изображении")
    plt.imshow(background, cmap='gray')

    # Эталон
    plt.subplot(325)
    plt.title("Эталон")
    plt.imshow(target, cmap='gray')

    plt.show()


def finding_triangle(image):
    """Ищет треугольник на изображении"""
    # Бинаризуем изображение

    # Гауссовый фильтр с бинаризацией Оцу
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    blur = np.pad(blur, pad_width=((2, 2), (1, 1)))
    # cor_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Применяем глобальную бинаризацию
    cor_image = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((7, 7), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((18, 18), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_CLOSE, kernel)

    return cor_image


def finding_circle(image):
    """Ищет круг на изображении"""
    kernel = np.ones((2, 2), np.uint8)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    cor_image = cv2.threshold(blur, 150, 255, cv2.THRESH_TOZERO_INV)[1]
    blur = cv2.GaussianBlur(cor_image, (5, 5), 0)
    cor_image = cv2.threshold(blur, 105, 255, cv2.THRESH_BINARY)[1]

    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((7, 7), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((24, 24), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((13, 13), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_CLOSE, kernel)

    return cor_image


def finding_background(image):
    """Ищет фон изображения"""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    blur = np.pad(blur, pad_width=((2, 2), (1, 1)))
    cor_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = np.ones((4, 4), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((6, 6), np.uint8)
    cor_image = cv2.morphologyEx(cor_image, cv2.MORPH_CLOSE, kernel)

    return cor_image


def area_on_model(model):
    """Выделяет объектные области"""

    # Выделенные объекты включают в себя также мелкие шумы
    # Избавляемся от них морфологическими операциями
    kernel = np.ones((2, 2))
    model_triangle = cv2.inRange(model, 155, 161)
    model_triangle = cv2.morphologyEx(model_triangle, cv2.MORPH_CLOSE, kernel)

    model_circle = cv2.inRange(model, 140, 146)
    model_circle = cv2.morphologyEx(model_circle, cv2.MORPH_OPEN, kernel)
    model_circle = cv2.morphologyEx(model_circle, cv2.MORPH_CLOSE, kernel)

    model_background = cv2.inRange(model, 92, 115)
    model_background = cv2.morphologyEx(model_background, cv2.MORPH_CLOSE, kernel)

    return model_triangle, model_circle, model_background


def object_box(object):
    """Помещает объект в коробку"""
    plot_one_img(object)
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    for i in range(0, len(object) - 1):
        if xmin == 0:
            for j in range(0, len(object[0]) - 1):
                if object[i, j] == 255:
                    xmin = i
                    break
        else:
            break

    for i in range(0, len(object[0]) - 1):
        if ymin == 0:
            for j in range(0, len(object) - 1):
                if object[j, i] == 255:
                    ymin = i
                    break
        else:
            break

    for i in range(len(object) - 1, 0, -1):
        if xmax == 0:
            for j in range(len(object[0]) - 1, 0, -1):
                if object[i, j] == 255:
                    xmax = i
                    break
        else:
            break

    for i in range(len(object[0]) - 1, 0, -1):
        if ymax == 0:
            for j in range(len(object) - 1, 0, -1):
                if object[j, i] == 255:
                    ymax = i
                    break
        else:
            break

    return xmin, ymin, xmax, ymax


def iou(xmin_mt, ymin_mt, xmax_mt, ymax_mt, xmin_t, ymin_t, xmax_t, ymax_t):
    """Функция оценки качества"""
    x1 = max(xmin_mt, xmin_t)
    x2 = min(xmax_mt, xmax_t)
    y1 = max(ymin_mt, ymin_t)
    y2 = min(ymax_mt, ymax_t)
    s1 = (x2 - x1) * (y2 - y1)  # S of intercept of rectangles
    s2 = (xmax_mt - xmin_mt) * (ymax_mt - ymin_mt)  # S of 1st rect
    s3 = (xmax_t - xmin_t) * (ymax_t - ymin_t)  # S of 2nd rect
    res = s1 / (s2 + s3 - s1)  # S of all area
    return res


image = initialize_image('images/noise.png')
model = initialize_image('images/gt.png')

# Ищем объекты
triangle = finding_triangle(image)
circle = finding_circle(image)
background = finding_background(image)

# Области объектов на эталоне
model_triangle, model_circle, model_background = area_on_model(model)

# Ищем коробки для изображений треугольников
xmin_mt, ymin_mt, xmax_mt, ymax_mt = object_box(model_triangle)
xmin_t, ymin_t, xmax_t, ymax_t = object_box(triangle)
# Вычисляем IoU
triangle_iou = iou(xmin_mt, ymin_mt, xmax_mt, ymax_mt, xmin_t, ymin_t, xmax_t, ymax_t)

# Ищем коробки для изображений кругов
xmin_mt, ymin_mt, xmax_mt, ymax_mt = object_box(model_circle)
xmin_t, ymin_t, xmax_t, ymax_t = object_box(circle)
# Вычисляем IoU
circle_iou = iou(xmin_mt, ymin_mt, xmax_mt, ymax_mt, xmin_t, ymin_t, xmax_t, ymax_t)

# Ищем коробки для изображений фонов(фон и так на всю картинку)
xmin_mt, ymin_mt, xmax_mt, ymax_mt = object_box(model_background)
xmin_t, ymin_t, xmax_t, ymax_t = object_box(background)
# Вычисляем IoU
background_iou = iou(xmin_mt, ymin_mt, xmax_mt, ymax_mt, xmin_t, ymin_t, xmax_t, ymax_t)

# Выводим значения IoU в консоль
print('triangle_IoU =', triangle_iou, 'circle_IoU =', circle_iou, 'background_IoU =', background_iou)

# Строим общий график
plot_images(image, triangle, circle, background, model)
