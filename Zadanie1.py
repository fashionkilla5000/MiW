from collections import Counter
import easygui
import pandas as pd
import os
import numpy as np
from numpy import nan
import math as m


def wczytajPlik():
    sciezka = easygui.fileopenbox()
    global data_name
    try:
        data_name = sciezka.split('\\')[3].split('.')[0]
        if ".data" in sciezka:
            return pd.read_csv(sciezka, header=None, na_values='?', sep=",")
        else:
            return pd.read_csv(sciezka, header=None, na_values='?', delim_whitespace=True)
    except:
        print("\nCzy napewno chodziło Ci o ten plik? Uruchom program ponownie i wybierz odpowiedni plik.")
        exit()


def czySymboliczne(df,cols):
    data = df.to_numpy()
    lista = []
    for y in range(cols-1):
        if type(data[0,y]) == str:
            lista.append(y)
    return lista


def daneSymboliczneNaLiczby(df, rows, cols):

    data = df.to_numpy()
    global G
    G = []
    for x in range(cols-1):
        if type(data[0, x]) is str:
            lista = []
            for y in range(rows):
                if type(data[y, x]) != float:
                    lista.append(data[y, x])
            counter = Counter(lista)
            i = len(counter.keys())
            for z in counter:
                s = 'Index kolumny: ' + str(x) + ' zamiana: ' + str(z) + ' na: ' + str(i)
                G.append(s)
                df = df.replace(to_replace=z, value=i)
                i -= 1
    return df


def normalizacja(df, nmin, nmax, rows, cols):
    global F
    F = []
    data = df.to_numpy()
    for y in range(cols-1):

        mini = min(data[:, y])
        maxi = max(data[:, y])

        s = "Kolumna: " + str(y+1) + ", Jej minimum: " + str(mini) + ", Jej maximum: " + str(maxi) + ","
        F.append(s)
        for x in range(rows):
            data[x, y] = (data[x, y]-mini) / (maxi-mini) * (nmax-nmin) + nmin
    return pd.DataFrame(data)


def zadanie1():

    global dane_przedEdycja, kopia
    dane_przedEdycja = wczytajPlik()

    global rows, cols
    rows, cols = dane_przedEdycja.shape

    A="Nazwa Dataset'u: "+data_name+"\n"+"Kolumna z klasą decyzyjną: "+str(cols)+"\n"
    B="Liczba wierszy: " + str(rows) + "\n"
    C="Liczba kolumn: " + str(cols) + "\n"

    global nmin, nmax
    nmin = int(input("Podaj dane do normalizacji(przedział dolny): "))
    nmax = int(input("Podaj dane do normalizacji(przedział górny): "))

    E = "Przedzial normalizacji: " + "[" + str(nmin) + "," + str(nmax) + "]" + "\n"

    if len(czySymboliczne(dane_przedEdycja,cols)) > 0:

        D="Dataset zawiera dane symboliczne, w kolumnach o indexach: "+str(czySymboliczne(dane_przedEdycja,cols))+"\n"

        dane = daneSymboliczneNaLiczby(dane_przedEdycja, rows, cols)
        dane = normalizacja(dane, nmin, nmax, rows, cols)

    else:

        D="Dataset nie zawiera danych symbolicznych\n"
        dane = normalizacja(dane_przedEdycja, nmin, nmax, rows, cols)

    print()

    global kopia
    kopia = dane

    with open(data_name + "_config.txt", 'w') as f:
        f.writelines(A+B+C+D+E)
        if 'G' in globals():
            for x in G:
                f.write(x + "\n")
        for x in F:
            f.write(x + "\n")


    with open(data_name+'_normalized.txt', 'w') as f:
        f.write(dane.to_string(header=False, index=False))

    os.startfile(data_name+'_normalized.txt')
    os.startfile(data_name+'_config.txt')


zadanie1()

######################################################################

def podaj_i_normalizuj_wpisane_wartosci(dane, ncols, nrows):
    wartosci = []
    print("Podaj wartosci do sklasyfikowania:")

    if len(czySymboliczne(dane, ncols)) > 0:
        print("(dataset zawiera dane symboliczne)\n")
        symboliczne = czySymboliczne(dane, ncols)

        for x in range(ncols - 1):
            if x in symboliczne:
                wartosci.append(str(input("[" + str(x) + "]" + "str: ")))
            else:
                wartosci.append(float(input("[" + str(x) + "]" + "float: ")))

        wartosci.append(np.nan)

        dane = dane.append(pd.DataFrame(wartosci).T, ignore_index=True)
        dane_nowe = daneSymboliczneNaLiczby(pd.DataFrame(dane), nrows, ncols)

    else:
        for x in range(ncols - 1):
            wartosci.append(float(input("[" + str(x) + "]" + "float: ")))

        wartosci.append(np.nan)

        dane_nowe = dane.append(pd.DataFrame(wartosci).T, ignore_index=True)

    znormalizowane = normalizacja(dane_nowe, nmin, nmax, nrows + 1, ncols)
    war_znormali = znormalizowane.to_numpy()[nrows][:]

    print("Podana próbka: \n", war_znormali)

    return war_znormali

def k_najb_probek(war_znormali, kopia, ncols, nrows, k, metryka,p):
    odl = []

    wynik = {
        "Parametr K": k,
        "Probka": list(war_znormali),
        "Sklasyfikowano": True,
        "Decyzja": ""
    }

    for x in range(nrows - 1):
        if metryka == 1:
            odl.append((kopia.to_numpy()[x, ncols - 1], metryka_euklides(kopia.to_numpy()[x], war_znormali)))
        if metryka == 2:
            odl.append((kopia.to_numpy()[x, ncols - 1], metryka_logarytm(kopia.to_numpy()[x], war_znormali)))
        if metryka == 3:
            odl.append((kopia.to_numpy()[x, ncols - 1], metryka_czebyszew(kopia.to_numpy()[x], war_znormali)))
        if metryka == 4:
            odl.append((kopia.to_numpy()[x, ncols - 1], metryka_minkowski(kopia.to_numpy()[x], war_znormali,p)))
        if metryka == 5:
            odl.append((kopia.to_numpy()[x, ncols - 1], metryka_manhattan(kopia.to_numpy()[x], war_znormali)))

    odl.sort(key=lambda tup: tup[1])
    odl = pd.DataFrame(odl).head(k)
    zlicz = Counter(odl.to_numpy()[:, 0])
    values = list(zlicz.values())
    if len(values) > 1:
        if values[0] == values[1]:
            wynik["Sklasyfikowano"] = False
            return wynik
    decyzja = zlicz.most_common(1)[0][0]

    wynik["Decyzja"] = decyzja
    return wynik

def najmniejsza_suma_odl(kopia, war_znormali, ncols, k, metryka,p):

    wynik = {
        "Parametr K": k,
        "Probka": list(war_znormali),
        "Sklasyfikowano": True,
        "Decyzja": ""
    }

    kopia.columns = range(ncols)
    decyzje = set(kopia.to_numpy()[:, ncols - 1])
    plus, minus = list(decyzje)[0], list(decyzje)[1]
    plusy, minusy = [], []
    for x in range(rows):
        if kopia.to_numpy()[x, ncols - 1] == plus:
            plusy.append(kopia.to_numpy()[x, :])
        if kopia.to_numpy()[x, ncols - 1] == minus:
            minusy.append(kopia.to_numpy()[x, :])

    plusy = pd.DataFrame(plusy)
    minusy = pd.DataFrame(minusy)

    odl_plusy = []
    odl_minusy = []

    for x in range(len(plusy.index) - 1):
        if metryka == 1:
            odl_plusy.append(
                (plusy.to_numpy()[x, len(plusy.columns) - 1], metryka_euklides(plusy.to_numpy()[x], war_znormali)))
        if metryka == 2:
            odl_plusy.append(
                (plusy.to_numpy()[x, len(plusy.columns) - 1], metryka_logarytm(plusy.to_numpy()[x], war_znormali)))
        if metryka == 3:
            odl_plusy.append(
                (plusy.to_numpy()[x, len(plusy.columns) - 1], metryka_czebyszew(plusy.to_numpy()[x], war_znormali)))
        if metryka == 4:
            odl_plusy.append(
                (plusy.to_numpy()[x, len(plusy.columns) - 1], metryka_minkowski(plusy.to_numpy()[x], war_znormali,p)))
        if metryka == 5:
            odl_plusy.append(
                (plusy.to_numpy()[x, len(plusy.columns) - 1], metryka_manhattan(plusy.to_numpy()[x], war_znormali)))
    for x in range(len(minusy.index) - 1):
        if metryka == 1:
            odl_minusy.append(
                (minusy.to_numpy()[x, len(plusy.columns) - 1], metryka_euklides(minusy.to_numpy()[x], war_znormali)))
        if metryka == 2:
            odl_minusy.append(
                (minusy.to_numpy()[x, len(plusy.columns) - 1], metryka_logarytm(minusy.to_numpy()[x], war_znormali)))
        if metryka == 3:
            odl_minusy.append(
                (minusy.to_numpy()[x, len(plusy.columns) - 1], metryka_czebyszew(minusy.to_numpy()[x], war_znormali)))
        if metryka == 4:
            odl_minusy.append(
                (minusy.to_numpy()[x, len(plusy.columns) - 1], metryka_minkowski(minusy.to_numpy()[x], war_znormali,p)))
        if metryka == 5:
            odl_minusy.append(
                (minusy.to_numpy()[x, len(plusy.columns) - 1], metryka_manhattan(minusy.to_numpy()[x], war_znormali)))

    odl_plusy.sort(key=lambda tup: tup[1])
    odl_plusy = pd.DataFrame(odl_plusy).head(k)

    odl_minusy.sort(key=lambda tup: tup[1])
    odl_minusy = pd.DataFrame(odl_minusy).head(k)

    suma_plusy = sum(odl_plusy.to_numpy()[:, 1])
    suma_minusy = sum(odl_minusy.to_numpy()[:, 1])

    decyzja = ""
    if suma_minusy < suma_plusy:
        wynik["Decyzja"] = str(odl_minusy.to_numpy()[0, 0])
    elif suma_minusy > suma_plusy:
        wynik["Decyzja"] = str(odl_plusy.to_numpy()[0, 0])
    else:
        wynik["Sklasyfikowano"] = False

    return wynik

def metryka_euklides(row1, row2):
    odleglosc = 0.0
    for i in range(len(row1) - 1):
        if np.isnan(row1[i]) or np.isnan(row2[i]):
            continue
        else:
            odleglosc += m.pow((row1[i] - row2[i]), 2)
    return m.sqrt(odleglosc)

def metryka_logarytm(row1, row2):
    odleglosc = 0.0
    for i in range(len(row1) - 1):
        if np.isnan(row1[i]) or np.isnan(row2[i]):
            continue
        else:
            odleglosc += m.fabs(m.log10(row1[i]) - m.log10(row2[i]))
    return odleglosc

def metryka_czebyszew(row1, row2):
    odleglosc = []
    for i in range(len(row1) - 1):
        if np.isnan(row1[i]) or np.isnan(row2[i]):
            continue
        else:
            odleglosc.append(m.fabs(row1[i] - row2[i]))
    return max(odleglosc)

def metryka_minkowski(row1, row2, p):
    odleglosc = 0.0
    for i in range(len(row1) - 1):
        if np.isnan(row1[i]) or np.isnan(row2[i]):
            continue
        else:
            odleglosc += m.pow(m.fabs(row1[i] - row2[i]), p)
    return m.pow(odleglosc, 1/p)

def metryka_manhattan(row1, row2):
    odleglosc = 0.0
    for i in range(len(row1) - 1):
        if np.isnan(row1[i]) or np.isnan(row2[i]):
            continue
        else:
            odleglosc += m.fabs(row1[i] - row2[i])
    return odleglosc


def zadanie2():
    p = 0
    dane = dane_przedEdycja
    print(dane)
    to_del = list(map(int, input("Podaj indexy kolumn do usunięcia (Jeśli nie chcesz usuwać kliknij Enter): ").split()))
    # to_del = [0,1,2,3,4,5,6,7,8,9,10,11]
    metryka = int(input("Podaj metrykę (1.euklides, 2.logarytm, 3.czebyszew, 4.minkowski, 5.manhatan): "))
    if metryka == 4:
        p = int(input("Podaj parametr p: "))
    for x in to_del:
        del dane[x]
        del kopia[x]
    nrows, ncols = dane.shape
    dane.columns = range(ncols)
    print(dane)

    opcja1 = int(
        input("Którym sposobem liczymy? (1. Dla podanych atrybutow; 2. jeden vs wiele): "))


    if opcja1 == 1:
        war_znormali = podaj_i_normalizuj_wpisane_wartosci(dane, ncols,nrows)

        k = int(input("Podaj parametr k:"))

        opcja1_1 = int(
            input("Którym sposobem liczymy? (1. K najblizszych probek; 2. Najmniejsza suma odleglosci w k probkach): "))

        if opcja1_1 == 1:
            wynik = k_najb_probek(war_znormali, kopia, ncols, nrows, k, metryka,p)
            print(wynik)

        if opcja1_1 == 2:
            wynik = najmniejsza_suma_odl(kopia, war_znormali, ncols, k,metryka,p)
            print(wynik)

    if opcja1 == 2:

        k = int(input("Podaj parametr k:"))

        poprawnie = 0
        blednie = 0
        nie_udalo_sie = 0

        opcja1_1 = int(
            input("Którym sposobem liczymy? (1. K najblizszych probek; 2. Najmniejsza suma odleglosci w k probkach): "))

        for x in range(nrows):
            print("Index Kolumny: ",x)
            decyzja = kopia.to_numpy()[x,ncols-1]
            war_znormali = kopia.to_numpy()[x,:-1]
            np.append(war_znormali,np.nan)


            if opcja1_1 == 1:
                wynik = k_najb_probek(war_znormali, kopia, ncols, nrows, k, metryka,p)

                if str(decyzja) == str(wynik["Decyzja"]):
                    poprawnie+=1
                elif wynik["Decyzja"] == "":
                    nie_udalo_sie+=1
                else:
                    blednie+=1

            if opcja1_1 == 2:
                wynik = najmniejsza_suma_odl(kopia, war_znormali, ncols, k,metryka,p)

                if str(decyzja) == str(wynik["Decyzja"]):
                    poprawnie+=1
                elif wynik["Decyzja"] == "":
                    nie_udalo_sie+=1
                else:
                    blednie+=1

        print("\nPoprawnie:", poprawnie, ",Błędnie: ", blednie, ",Nie udało się sklasyfikować: ", nie_udalo_sie,
              "\nWspółczynnik poprawności: ", int((poprawnie/(blednie+nie_udalo_sie+poprawnie))*100), '%')

zadanie2()