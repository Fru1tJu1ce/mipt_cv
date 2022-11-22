import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def initialize_picture(path: str):
    """Инициализирует картинку"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def crop(image):
    """Убирает черную область"""
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def plot_image(image, cmap=None):
    plt.figure(figsize=(16, 9))

    plt.title("Сшитая картинка")
    plt.imshow(image, cmap)

    plt.show()


def _stitch_images(image, img, matches, kp1, kp2):
    """Сшивает картинки"""
    if len(matches) >= 2:
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        try:
            H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        except:
            return 1

    else:
        return 1

    try:
        im_out = cv2.warpPerspective(image, H, (image.shape[1] + img.shape[1], img.shape[0]))
        im_out[0:img.shape[0], 0:img.shape[1]] = img
        return im_out
    except:
        return 1


def match_images(images: list) -> list:
    """Сопоставляет картинки"""
    hyp_params = dict(
        nfeatures=100,
        nOctaveLayers=5,
        contrastThreshold=0.03,
        edgeThreshold=10,
        sigma=1.6)  # hyp params
    detector = cv2.SIFT_create(**hyp_params)

    rest_images = images.copy()
    stitched_images = list()
    for image in images:
        for img in rest_images:
            # matching algorithm
            keypoints1, desc1 = detector.detectAndCompute(image, None)
            keypoints2, desc2 = detector.detectAndCompute(img, None)

            # init match algo
            FLANN_INDEX_KDTREE = 2
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)

            # apply threshold rule
            ratio_thresh = 0.7
            good = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

            # draw stitched image
            im_out = _stitch_images(image, img, good, keypoints1, keypoints2)
            if type(im_out) != int:

                # deleting black area
                im_out = crop(im_out)
                if im_out.shape[1] > img.shape[1]:
                    stitched_images.append(im_out)

    return stitched_images


def list_of_imgs(imgs_dir: str) -> list:
    """Возвращает список картинок из директории"""
    images = list()
    # Перебираем картинки из директории с картинками
    for i, img in enumerate(os.listdir(imgs_dir)):
        path = os.path.join(imgs_dir, img)
        image = initialize_picture(path)
        images.append(image)
    return images


def crop_stitched_images(images) -> list:
    """Убирает черную область из каждого сшитого изображения."""
    cropped = list()
    for img in images:
        new_img = crop(img)
        cropped.append(new_img)

    return cropped


if __name__ == '__main__':
    imgs_dir = 'images/su_fighter_shuffle'
    images = list_of_imgs(imgs_dir)

    # Сшиваем картинки, пока не останется одна большая
    stitched_images = list()
    while len(stitched_images) != 1:
        stitched_images = match_images(images)
        images = crop_stitched_images(stitched_images)

    # Выводим результат
    plot_image(images[0])
