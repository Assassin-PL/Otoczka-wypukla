from Klasy import *

# Ustawienia ilości figur do wygenerowania
ilosc_figur = 20
#konwersja pliku svg do zmiennej
path_png = "tiger.png"

width_tygrys, height_tygrys = 50, 50

max_dimension = (0, 800)

tygrys = cv2.imread(path_png)

tygrys.shape

# Sprawdź, czy wczytanie obrazu się powiodło
if tygrys is not None:
    # Wyświetl obraz
    print("dziala")
else:
    print("Wystąpił błąd podczas wczytywania pliku.")

nazwa = "kwadrat"

# Generowanie losowych figur
figury = [Figura(np.random.uniform(max_dimension[0], max_dimension[1]), np.random.uniform(max_dimension[0], max_dimension[1]), nazwa) for _ in range(ilosc_figur)]

# Przygotowanie danych do wykresu
srodki_x = [figura.srodekCiezkosci.x for figura in figury]
srodki_y = [figura.srodekCiezkosci.y for figura in figury]

punkty = [figura.zwroc_wiercholki() for figura in figury]

punkty_linii_x = np.array([[punkt.x for punkt in figura] + [figura[0].x] for figura in punkty])
punkty_linii_y = np.array([[punkt.y for punkt in figura] + [figura[0].y] for figura in punkty])

zbior_x = splaszcz_tablice(punkty_linii_x)
zbior_y = splaszcz_tablice(punkty_linii_y)

# Algorytm Grahama dla środków ciężkości
# convex_hull = graham_scan(np.column_stack((srodki_x, srodki_y)))
convex_hull = graham_scan(np.column_stack((zbior_x, zbior_y)))

# Rysowanie wykresu
# plt.scatter(srodki_x, srodki_y, label='Środki ciężkości')
plt.plot(punkty_linii_x.T, punkty_linii_y.T, color='gray', linestyle='-', alpha=0.5)

# Dodatkowe ustawienia wykresu
plt.title('Środki ciężkości i linie ' + nazwa + 'ów ')
plt.xlabel('Współrzędna x')
plt.ylabel('Współrzędna y')

# Rysowanie otoczki wypukłej dla środków ciężkości
# draw_convex_hull(np.column_stack((srodki_x, srodki_y)), convex_hull)
draw_convex_hull_with_image(np.column_stack((srodki_x, srodki_y)), convex_hull, path_png)
