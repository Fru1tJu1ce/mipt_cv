import cv2
import numpy as np
import matplotlib.pyplot as plt


def determine_wall_thickness(test_image) -> int:
    """Вычисляет толщину стен."""
    counter = 0
    for i in range(len(test_image)):
        if (test_image[i][i] == np.array([0, 0, 0])).all():
            counter += 1
        else:
            break
    return counter


def determine_void_wall_thickness(test_image):
    """Возвращает толщину стены и ширину клетки пути."""
    wall_thickness = determine_wall_thickness(test_image)
    counter = 0
    for i in range(wall_thickness, len(test_image)):
        if (test_image[i][i] == np.array([255, 255, 255])).all():
            counter += 1
        else:
            break
    return wall_thickness, counter


def calculating_maze_size(test_image, wall, void):
    """Вычисляет размер лабиринта."""
    x_size = int((len(test_image[0]) - wall) / (wall + void))
    y_size = int((len(test_image) - wall) / (wall + void))
    return x_size, y_size


def convert_maze(test_image, wall, void, x_size, y_size):
    """Превращает лабиринт в массив из 0 и 1."""
    maze_in_array = np.zeros(((x_size * 2 + 1), (y_size * 2 + 1)), dtype=int)
    step = int((wall + void) / 2)
    maze_array_counter_x = 0
    for i in range(0, len(test_image) - wall + 1, step):
        maze_array_counter_y = 0
        for j in range(0, len(test_image[0]) - wall + 1, step):
            if (test_image[i][j] == np.array([0, 0, 0])).all():
                maze_in_array[maze_array_counter_x][maze_array_counter_y] = 1
            maze_array_counter_y += 1
        maze_array_counter_x += 1
    return maze_in_array


def input_start_end_point():
    """Введение начальной точки."""
    print('Стандартные координаты точек:\nначало - 1 19\nконец - 39 21')
    start_point = tuple(map(int, input("Введите координаты начальной точки через пробел: ").split()))
    end_point = tuple(map(int, input("Введите координаты конечной точки через пробел: ").split()))
    return start_point, end_point


def saving_new_maze_png(maze_in_array):
    """Сохраняет лабиринт в ужатом формате."""
    new_maze = np.zeros((len(maze_in_array), len(maze_in_array[0]), 3), dtype=np.uint8)
    for i in range(len(maze_in_array)):
        for j in range(len(maze_in_array[0])):
            if maze_in_array[i][j] == 1:
                new_maze[i][j] = np.array([0, 0, 0], dtype=np.uint8)
            elif maze_in_array[i][j] == 0:
                new_maze[i][j] = np.array([255, 255, 255], dtype=np.uint8)

    cv2.imwrite('new_maze.png', new_maze)


def saving_maze_with_way(maze_in_array, array_of_path):
    """Сохраняет лабиринт с решением."""
    new_maze = np.zeros((len(maze_in_array), len(maze_in_array[0]), 3), dtype=np.uint8)
    for i in range(len(maze_in_array)):
        for j in range(len(maze_in_array[0])):
            if maze_in_array[i][j] == 1:
                new_maze[i][j] = np.array([0, 0, 0], dtype=np.uint8)
            elif maze_in_array[i][j] == 0:
                new_maze[i][j] = np.array([255, 255, 255], dtype=np.uint8)
    for value in array_of_path:
        new_maze[value[0]][value[1]] = np.array([0, 255, 0], dtype=np.uint8)

    # Вывод решенного лабиринта
    fig, axs = plt.subplots(1, 1, figsize=(8, 7))
    axs.imshow(new_maze)
    axs.axis('off')
    plt.show()
    i = input('press return to end')
    #cv2.imwrite('new_maze_completed.png', new_maze)


def make_step(k, matrix, maze_in_array):
    """Шагаем в разные стороны от исходной точки."""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == k:
                if (i > 0) and (matrix[i - 1][j] == 0) and (maze_in_array[i - 1][j] == 0):
                    matrix[i - 1][j] = k + 1
                if (j > 0) and (matrix[i][j - 1] == 0) and (maze_in_array[i][j - 1] == 0):
                    matrix[i][j - 1] = k + 1
                if (i < len(matrix) - 1) and (matrix[i + 1][j] == 0) and (maze_in_array[i + 1][j] == 0):
                    matrix[i + 1][j] = k + 1
                if (j < len(matrix[i]) - 1) and (matrix[i][j + 1] == 0) and (maze_in_array[i][j + 1] == 0):
                    matrix[i][j + 1] = k + 1


def finding_ways_matrix(maze_in_array, start_point, end_point):
    """Ищет пути из лабиринта."""
    matrix_for_counting = np.zeros((len(maze_in_array), len(maze_in_array[0])), dtype=int)
    matrix_for_counting[start_point[0]][start_point[1]] = 1
    k = 0
    while matrix_for_counting[end_point[0]][end_point[1]] == 0:
        k += 1
        make_step(k, matrix_for_counting, maze_in_array)
    return matrix_for_counting


def finding_way_backwards(matrix, end_point):
    """Ищет кратчайший путь из последней точки в начало."""
    i, j = end_point[0], end_point[1]
    k = matrix[i][j]
    array_of_path = [(i, j)]
    while k > 1:
        if i > 0 and matrix[i - 1][j] == k - 1:
            i, j = i - 1, j
            array_of_path.append((i, j))
            k -= 1
        elif j > 0 and matrix[i][j - 1] == k - 1:
            i, j = i, j - 1
            array_of_path.append((i, j))
            k -= 1
        elif i < len(matrix) - 1 and matrix[i + 1][j] == k - 1:
            i, j = i + 1, j
            array_of_path.append((i, j))
            k -= 1
        elif j < len(matrix[i]) - 1 and matrix[i][j + 1] == k - 1:
            i, j = i, j + 1
            array_of_path.append((i, j))
            k -= 1
    return array_of_path


def find_way_from_maze():
    """Ищет путь из лабиринта. Закомментированные строки позволяют сохранить некоторые
    этапы программы в файлы для наглядности."""
    # Инициализация картинки, расчет толщины стены и ширины клетки пути
    test_image = cv2.imread('20_by_20_orthogonal_maze.png')
    wall, void = determine_void_wall_thickness(test_image)

    # Расчет размеров лабиринта, ввод начальной и конечной точки
    x_size, y_size = calculating_maze_size(test_image, wall, void)
    start_point, end_point = input_start_end_point()  # 1 19 и 39 21

    # Ужимание лабиринта (толщина стенки = 1 пиксель, ширина пути = 1 пиксель),
    # сохранение ужатой версии в файл 'new_maze.png'
    maze_in_array = convert_maze(test_image, wall, void, x_size, y_size)
    # np.savetxt("maze_in_array.csv", maze_in_array, fmt="%d", delimiter=',')
    # saving_new_maze_png(maze_in_array)

    # Расчет пути решения лабиринта
    matrix = finding_ways_matrix(maze_in_array, start_point, end_point)
    # np.savetxt("way_matrix.csv", matrix, fmt="%d", delimiter=',')
    array_of_path = finding_way_backwards(matrix, end_point)

    # Отрисовка пути в новом файле 'new_maze_completed.png', если раскомментировать кусок в функции
    saving_maze_with_way(maze_in_array, array_of_path)


find_way_from_maze()
