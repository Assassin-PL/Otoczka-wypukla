import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from numpy.random import rand
import cv2
from scipy.ndimage import rotate

def init():
    return []


def twierdzenie_cosinusow(a, b, c):
    return math.acos((a ** 2 + b ** 2) / (c ** 2 - 2 * a * b))


def splaszcz_tablice(tablica_2d):
    """
    Spłaszcza tablicę 2D do tablicy 1D.

    Parameters:
    - tablica_2d (list): Tablica dwuwymiarowa.

    Returns:
    - list: Spłaszczona tablica jednowymiarowa.
    """
    return [element for wiersz in tablica_2d for element in wiersz]


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def graham_scan(points):
    n = len(points)
    if n < 3:
        return points

    pivot = min(range(n), key=lambda i: (points[i][1], points[i][0]))

    sorted_points = sorted(range(n), key=lambda i: (np.arctan2(points[i][1] - points[pivot][1],
                                                               points[i][0] - points[pivot][0]),
                                                    np.linalg.norm(np.array(points[i]) - np.array(points[pivot]))))

    hull = [sorted_points[0], sorted_points[1]]
    for i in range(2, n):
        while len(hull) > 1 and orientation(points[hull[-2]], points[hull[-1]], points[sorted_points[i]]) != 2:
            hull.pop()
        hull.append(sorted_points[i])

    return [points[i] for i in hull]


def oblicz_punkty(kat, R):
    x = R * np.cos(np.radians(kat))
    y = R * np.sin(np.radians(kat))
    return x, y


def draw_convex_hull(points, convex_hull):
    plt.scatter(points[:, 0], points[:, 1], marker='o', label='Punkty środka ciężkości figury')
    plt.scatter(np.array(convex_hull)[:, 0], np.array(convex_hull)[:, 1], marker='o', color='red',
                label='Punkt otoczki wypukłej')
    plt.plot(np.array(convex_hull)[:, 0], np.array(convex_hull)[:, 1], 'r--', label='Otoczka wypukła')

    # Łącz punkt początkowy i końcowy otoczki wypukłej
    plt.plot([convex_hull[0][0], convex_hull[-1][0]], [convex_hull[0][1], convex_hull[-1][1]], 'r--')

    plt.title('Otoczka wypukła za pomocą algorytmu Grahama')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.legend()
    plt.show()

def draw_convex_hull_with_image(points, convex_hull, image_path):
    # Oblicz środek ciężkości
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])

    # Odczytaj obraz
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Wczytaj obrazek z przezroczystością

    # Wczytaj tylko odcień szarości i kanał alpha


    # gray_alpha_with_color = np.concatenate([alpha_channel, np.ones_like(alpha_channel[:, :, :1]) * 255], axis=-1)

    # Stwórz wykres
    fig, ax = plt.subplots()

    # Rysuj obrazki prezentu dla każdego punktu
    for x, y in zip(points[:, 0], points[:, 1]):
        rotation_angle = np.random.uniform(0, 180)

        rotated_image = rotate(image, rotation_angle, reshape=False)

        alpha_channel = rotated_image[:, :, 3]

        inverted_image = 255 - alpha_channel
        imagebox = OffsetImage(inverted_image, zoom=0.5, cmap='gray', norm=None, interpolation=None, origin=None, filternorm=1, filterrad=4.0, resample=True)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)

    # Rysuj otoczkę wypukłą
    plt.scatter(points[:, 0], points[:, 1], marker='o', label='Punkty środka ciężkości figury')
    plt.scatter(np.array(convex_hull)[:, 0], np.array(convex_hull)[:, 1], marker='o', color='red',
                label='Punkt otoczki wypukłej')
    plt.plot(np.array(convex_hull)[:, 0], np.array(convex_hull)[:, 1], 'r--', label='Otoczka wypukła')

    # Łącz punkt początkowy i końcowy otoczki wypukłej
    plt.plot([convex_hull[0][0], convex_hull[-1][0]], [convex_hull[0][1], convex_hull[-1][1]], 'r--')

    # Ustawienia wykresu
    plt.title('Otoczka wypukła za pomocą algorytmu Grahama z obrazkami prezentu (szarość + alpha)')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.legend()

    # Wyświetl wykres
    plt.grid(True)
    plt.show()

class Punkt2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, drugi):
        return Punkt2D(self.x + drugi.x, self.y + drugi.y)

    def __sub__(self, drugi):
        return Punkt2D(self.x - drugi.x, self.y - drugi.y)

    def __eq__(self, drugi):
        return self.x == drugi.x and self.y == drugi.y


class Figura(Punkt2D):
    def __init__(self, x, y, typ_figury):
        super().__init__(x, y)
        self.__R = 50 / np.sqrt(2)
        self.alpha = np.random.uniform(0, 360)  # zaminiac typ zmiennej na prywatna zapamietac
        self.beta = np.random.uniform(0, 90)
        self.gamma = np.random.uniform(0, 45)
        self.a = np.random.uniform((0, 10))
        self.b = 2
        self.__wierzcholek = []
        self.srodekCiezkosci = Punkt2D(self.x, self.y)
        if typ_figury == "trojkat":
            self.__wierzcholek.append(
                self.srodekCiezkosci + Punkt2D(oblicz_punkty(90, self.__R)[0], oblicz_punkty(90, self.__R)[1]))
            self.__wierzcholek.append(
                self.srodekCiezkosci + Punkt2D(oblicz_punkty(210, self.__R)[0], oblicz_punkty(210, self.__R)[1]))
            self.__wierzcholek.append(
                self.srodekCiezkosci + Punkt2D(oblicz_punkty(330, self.__R)[0], oblicz_punkty(330, self.__R)[1]))
        if typ_figury == "kwadrat":
            self.alpha = 0
            # self.beta = 22.5
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(45 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(45 + self.alpha, self.__R)[1]))
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(135 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(135 + self.alpha, self.__R)[1]))
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(225 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(225 + self.alpha, self.__R)[1]))
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(315 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(315 + self.alpha, self.__R)[1]))
        if typ_figury == "pieciokat":
            # dlugosc_bokow = oblicz_punkty( twierdzenie_cosinusow(self.__R, self.__R, self.a),self.__R)
            self.alpha = np.random.uniform(0, 90)
            self.beta = np.random.uniform(0, 90)
            self.__R = np.random.uniform(self.b, 10)
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(45 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(45 + self.alpha, self.__R)[1]))
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(135 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(135 + self.alpha, self.__R)[1]))
            if 135 + self.alpha + self.beta < 225 + self.alpha: self.__wierzcholek.append(
                self.srodekCiezkosci + Punkt2D(oblicz_punkty(135 + self.alpha + self.beta, self.__R)[0] - self.b,
                                               oblicz_punkty(135 + self.alpha + self.beta, self.__R)[1] - self.b))
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(225 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(225 + self.alpha, self.__R)[1]))
            if 135 + self.alpha + self.beta > 225 + self.alpha: self.__wierzcholek.append(
                self.srodekCiezkosci + Punkt2D(oblicz_punkty(135 + self.alpha + self.beta, self.__R)[0] - self.b,
                                               oblicz_punkty(135 + self.alpha + self.beta, self.__R)[1] - self.b))
            self.__wierzcholek.append(self.srodekCiezkosci + Punkt2D(oblicz_punkty(315 + self.alpha, self.__R)[0],
                                                                     oblicz_punkty(315 + self.alpha, self.__R)[1]))
        if typ_figury == "dziwny trojkat":
            pass

    def zwroc_wiercholki(self):
        return tuple(self.__wierzcholek)

    def update(self, x, y):

        przesuniecie = Punkt2D(x, y)
        self.srodekCiezkosci += przesuniecie
        for punkt in self.__wierzcholek:
            punkt += przesuniecie

        pass

    def draw(self):
        plt.scatter([punkt.x for punkt in self.__wierzcholek],
                    [punkt.y for punkt in self.__wierzcholek], marker='o')
        plt.plot([punkt.x for punkt in self.__wierzcholek] + [self.__wierzcholek[0].x],
                 [punkt.y for punkt in self.__wierzcholek] + [self.__wierzcholek[0].y], 'k-')
