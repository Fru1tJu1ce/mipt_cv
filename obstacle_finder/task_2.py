import cv2
import matplotlib.pyplot as plt


def initialize_pic(image):
    """Преобразует картинку в формат кодов значений пикселей в зависимости от цвета."""
    array_of_image = cv2.imread(image)
    array_of_image = cv2.cvtColor(array_of_image, cv2.COLOR_BGR2RGB)
    return array_of_image


def intervals_of_roads(road_area):
    """Вычисляет количество дорог."""
    i = 0
    array_of_intervals = []
    while i != (len(road_area[0]) - 1):
        if road_area[len(road_area) - 1][i] == 0:
            i += 1
            continue
        else:
            start = i
            while road_area[len(road_area) - 1][i] == 255:
                i += 1
            array_of_intervals.append([start, i])
    return array_of_intervals


def checking_obstacles(red_area, array_of_roads):
    """Ищет препятствия"""
    for i in range(len(red_area)):
        for j in range(len(red_area)):
            if red_area[i][j] == 0:
                continue
            else:
                for value in array_of_roads:
                    if j >= value[0] and j <= value[1]:
                        array_of_roads.remove(value)


def move_or_dont(car_area, array_of_roads):
    """Проверяет, нужно ли перестраиваться или нет."""
    for i in range(len(car_area)):
        for j in range(len(car_area[0])):
            if car_area[i][j] == 0:
                continue
            else:
                for value in array_of_roads:
                    if j >= value[0] and j <= value[1]:
                        array_of_roads.remove(value)
                break
        break
    return array_of_roads


def printing_pictures(picture, red_area, road_area, car_area):
    """Строит 'серую' карту препятствий, машины и дорог"""
    fig, m_axs = plt.subplots(2, 2, figsize = (12, 9))
    (ax1, ax2), (ax3, ax4) = m_axs

    ax1.set_title('Исходная картинка', fontsize=15)
    ax1.imshow(picture)

    ax2.set_title('Только препятствия', fontsize=15)
    ax2.imshow(red_area, cmap='gray')

    ax3.set_title('Только машина', fontsize=15)
    ax3.imshow(car_area, cmap='gray')

    ax4.set_title('Только дорога', fontsize=15)
    ax4.imshow(road_area, cmap='gray')
    plt.show()


#red from [200 0 0] to [255 60 10]
#grey from [200 200 200] to [220 220 220]
#yellow from [240 240 110] to [255 255 170]

def find_road_number(path='image_01.jpg'):
    """Выводит номер дороги, на которую нужно перестроиться, если перестроение необходимо"""
    # Инициализация фотографии
    picture = initialize_pic(path)
    # ЧБ массивы препятствий, дорог и машины
    red_area = cv2.inRange(picture, (200, 0, 0), (255, 60, 10))
    road_area = cv2.inRange(picture, (200, 200, 200), (240, 240, 240))
    car_area = cv2.inRange(picture, (0, 0, 150), (140, 140, 255))

    # Вычисление интервалов пикселей дорог, проверка на наличие препятствий на них
    array_of_roads = intervals_of_roads(road_area)
    saving_array = array_of_roads[:]
    checking_obstacles(red_area, array_of_roads)

    # Номер пустой дороги - индекс пересечения значения интервала для пустой дороги с массивом интервалов для дорог
    num_of_road = saving_array.index(array_of_roads[0])
    # ЧБ карта элементов на картинке
    #printing_pictures(picture, red_area, road_area, car_area)
    # Проверка на нужность перестроения в другой ряд
    checking_array = move_or_dont(car_area, array_of_roads)
    if not checking_array:
        print('Перестраиваться не нужно.')
    else:
        print(f'Нужно перестроиться на дорогу номер {num_of_road}.')

    return num_of_road

road_number = find_road_number()