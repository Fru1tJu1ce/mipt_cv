import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import time

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.figsize'] = (10.0, 8.0)


def initialize_image(image='images/dog.jpg'):
    """Инициализирует картинку"""
    image = cv2.imread(image, 0)
    return image


def show_images(image, padded_image, result):
    """Выводит изображение картинки"""
    fig, m_axs = plt.subplots(1, 3, figsize=(15, 5))
    ax_1, ax_2, ax_3 = m_axs

    ax_1.set_title('Исходное изображение')
    ax_1.set_axis_off()
    ax_1.imshow(image)

    ax_2.set_title('Padded изображение')
    ax_2.imshow(padded_image)
    ax_2.set_axis_off()

    ax_3.set_title('Что должно получиться')
    ax_3.imshow(result)
    ax_3.set_axis_off()

    plt.show()


def show_one_image(image):
    """Выводит одну картинку"""
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def conv_nested(image, kernel):
    """Convolution filter"""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = np.pad(image, Hk // 2)

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[i, j] += image[i + k, j + l] * kernel[Hk - 1 - k, Wk - 1 - l]

    return out


def conv_fast(image, kernel):
    """Fast convolution"""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    flipped_kernel = np.flip(kernel)
    for i in range(Hi - Hk + 1):
        for j in range(Wi - Wk + 1):
            out[i, j] = np.sum(image[i: i + Hk, j: j + Wk] * flipped_kernel)
    return out


def zero_pad(image, pad_height, pad_width):
    """Паддит нули по вертикали и горизонтали"""
    image = np.pad(image, pad_width=((pad_height, pad_height), (pad_width, pad_width)))
    return image


image = initialize_image()

kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

out = conv_nested(image, kernel)
# out = cv2.filter2D(img, -1, kernel)

# Plot original image
plt.subplot(221), plt.imshow(image), plt.title('Original'), plt.axis('off')

# Plot your convolved image
plt.subplot(223), plt.imshow(out), plt.title('Convolution'), plt.axis('off')

# Plot what you should get
solution_img = cv2.imread('images/convoluted_dog.jpg', 0)
plt.subplot(224), plt.imshow(solution_img), plt.title('What you should get'), plt.axis('off')
plt.show()

pad_height = 40
pad_width = 20
padded_image = zero_pad(image, pad_height, pad_width)
result_img = cv2.imread('images/padded_dog.jpg', 0)
show_images(image, padded_image, result_img)

kernel = 1/273 * np.array([[1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]])
t0 = time()
out_fast = conv_fast(padded_image, kernel)
t1 = time()
out_nested = conv_nested(padded_image, kernel)
t2 = time()

# Compare the running time of the two implementations
print("conv_nested: took %f seconds." % (t2 - t1))
print("conv_fast: took %f seconds." % (t1 - t0))

# Plot conv_nested output

plt.subplot(1, 2, 1)
plt.imshow(out_nested)
plt.title('conv_nested')
plt.axis('off')

# Plot conv_fast output
plt.subplot(1, 2, 2)
plt.imshow(out_fast)
plt.title('conv_fast')
plt.axis('off')
plt.show()