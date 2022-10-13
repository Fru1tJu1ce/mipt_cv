import matplotlib.pyplot as plt
import numpy as np
import cv2
import operator


def initialize_image(path='images/test_image_00.jpg'):
    """Инициализирует изображение и преобразует его в RGB"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plot_image(image):
    """Выводит изображение, гистограмму и определяет принадлежность к пустыне или лесу"""
    plt.figure(figsize=(15, 9))

    # Строим гистограмму
    plt.subplot(212)
    plt.title('Гистограмма изображения')
    color = ('r', 'g', 'b')
    data = {}
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [246], [10, 256])
        hist = np.concatenate((np.zeros((10, 1), dtype=np.float32), hist))
        data[col] = np.argmax(hist)
        plt.plot(hist, color=col)
        plt.xlim([10, 256])

    # Выводим исходную картинку с результатом лес/пустыня
    result = forest_or_desert(data)
    plt.subplot(211).set_title(result)
    plt.subplot(211).imshow(image)

    plt.show()


def forest_or_desert(data):
    """Определяет лес или пустыню по максимальному значению в словаре"""
    key = max(data.items(), key=operator.itemgetter(1))[0]
    if key == 'r':
        return 'Это - пустыня'
    elif key == 'g':
        return 'Это - лес'


image = initialize_image('images/test_image_00.jpg')
plot_image(image)
